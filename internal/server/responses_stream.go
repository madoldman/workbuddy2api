// responses_stream.go 上游 chat SSE 流 → OpenAI Responses 事件流的翻译器。
//
// 事件序列（SSE：`event: <type>\ndata: {...}\n\n`，Codex 严格依赖）：
//
//	response.created → [reasoning 收口三连] → message item（added / part.added /
//	output_text.delta* / part.done / output_text.done / item.done）→ function_call
//	items（added / arguments.delta* / arguments.done / item.done）→ response.completed。
//
// 设计决议：reasoning **不发增量 delta**，在首个正文/工具分片到达（或流结束）时
// 一次性发 added + reasoning_summary_text.done（完整文本）——Codex 不依赖 reasoning
// 的增量形态，只要 item 结构完整；该方案把思维链从「逐帧状态机」简化为「缓冲 + 一次性
// 收口」，输出序号分配也随之自然（reasoning 先收口则占 0 号位）。
//
// 事件族选 reasoning_summary_text.* 而非 reasoning_text.* 的原因：实测 Codex 只消费
// summary 形态（带 summary_index；reasoning_text.delta/done 不进 UI）。一次性收口方案
// 只换事件名，状态机不变。缓冲 64KB 封顶（按剩余额度截断，单帧超限同样生效）防异常
// 上游把网关内存打爆；正文缓冲 2MB 封顶同理。
//
// response.created 在写出 SSE 头后**立即**发出（不等首个有效帧）——空流/错误帧场景
// 客户端也能先拿到 response 对象再收 failed，与官方「created 恒为首个事件」一致。
package server

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strings"
	"time"

	"workbuddy2api/internal/auth"
	"workbuddy2api/internal/logfmt"
	"workbuddy2api/internal/session"
	"workbuddy2api/internal/upstream"
)

// reasoningCap 思维链缓冲封顶（64KB）：一次性收口方案下 reasoning 全文驻留内存，
// 异常上游的无限 reasoning 流不能拖垮网关。截断按剩余额度生效（单帧超限同样裁剪），
// 静默发生——对 Codex 只是少一段思维链展示，不破坏事件结构。
const reasoningCap = 64 * 1024

// textCap 正文缓冲封顶（2MB）：appendText 与 appendReasoning 同理，异常上游的
// 无限 content 流不能拖垮网关。超限后静默丢弃后续增量（事件流结构不破坏）。
const textCap = 2 << 20

// errResponsesEmptyStream 上游 200 但 0 有效 SSE 帧（与 upstream.errEmptyStream
// 同语义的包内哨兵——upstream 的哨兵未导出，errors.Is 无法跨包构造）。
var errResponsesEmptyStream = errors.New("upstream stream contained no valid data events")

// responsesToolCallState 单个 tool call 的流式累积状态（按上游 index 归位）。
type responsesToolCallState struct {
	outIdx int    // Responses 事件流的 output_index（分配即固定）
	itemID string // "fc_" + 32hex，item 级 id
	callID string // 上游 tool_call id（Codex 回传 function_call_output.call_id 的配对键）
	name   string // 函数名（added/done 全量出现，无 name delta 事件）
	args   strings.Builder
	added  bool // output_item.added 已发
}

// responsesStreamTranslator 流式翻译状态机。
type responsesStreamTranslator struct {
	w           http.ResponseWriter
	fl          http.Flusher
	responseID  string
	model       string
	createdAt   int64
	sequence    int64 // 事件序号（全流递增，Codex 按序消费）
	created     bool  // response.created 已发
	failed      bool  // 已发 response.failed（此后不再收口/完结）
	validFrames int

	nextOutputIndex int

	// reasoning：缓冲 + 一次性收口（见文件头设计决议）。
	reasoningBuf     strings.Builder
	reasoningFlushed bool
	reasoningItemID  string
	reasoningOutIdx  int

	// 正文 message item 状态（全流单 item——chat SSE 的 delta.content 天然单路）。
	textStarted bool
	textItemID  string
	textOutIdx  int
	textBuf     strings.Builder

	// tool calls：index → 状态；toolOrder 保序；idIndex 缺 index 时按 id 归位
	// （与 upstream/sse.go Aggregate 的 mergeToolCallsChunk 同一归位策略）。
	toolCalls map[int]*responsesToolCallState
	toolOrder []int
	idIndex   map[string]int
	toolSeq   int

	usage        map[string]any
	finishReason string
	// sawUpstreamDone 上游是否显式发过 data: [DONE]：截断判定输入（与 chat 非流式
	// Aggregate 同口径——EOF 收尾但未发 DONE 视为流被截断，残缺 tool 参数不下发）。
	sawUpstreamDone bool
	// droppedTruncatedTool 是否有 tool call 因残缺参数被剔除（P1-1）：命中即把
	// completed.status 收敛为 incomplete——参数残缺多半来自 length 截断或断流，
	// 报 completed 会诱导 Codex 把脏轮次当成功继续。
	droppedTruncatedTool bool
	textTruncated        bool

	// toolMap namespace 工具映射表（拼接名 → 真实身份，ResponsesToChat 处理请求
	// 时生成并经 responsesPipeSink 下沉）：发射 function_call item 前按其回译
	// name/namespace——Codex 按 namespace 字段路由到 multi_agent 处理器，拼接名
	// 直透会让它找不到工具。nil = 无 namespace 工具，name 原样、namespace 省略。
	toolMap map[string]responsesToolIdentity
}

// responsesToolCallIdentity 按 toolMap 回译工具调用名：命中映射表返回
// (真实名, 命名空间, true)；未命中返回 (拼接名原样, "", false)——普通工具与
// 无 namespace 请求全走后者的零回归路径。
func (t *responsesStreamTranslator) responsesToolCallIdentity(joined string) (string, string, bool) {
	if id, hit := t.toolMap[joined]; hit {
		return id.Name, id.Namespace, true
	}
	return joined, "", false
}

// newResponsesStreamTranslator 构造翻译器（未写任何字节；SSE 头在 run 时设置）。
func newResponsesStreamTranslator(w http.ResponseWriter, model string) *responsesStreamTranslator {
	fl, _ := w.(http.Flusher)
	return &responsesStreamTranslator{
		w:          w,
		fl:         fl,
		responseID: "resp_" + session.NewMessageID(),
		model:      model,
		createdAt:  time.Now().Unix(),
		toolCalls:  map[int]*responsesToolCallState{},
		idIndex:    map[string]int{},
	}
}

// emit 写出一个 Responses 事件（event 行 + data 行 + 空行）并 flush。
// sequence_number 由本函数统一分配递增——官方事件流每事件必带，Codex 按序校验。
func (t *responsesStreamTranslator) emit(eventType string, data map[string]any) error {
	t.sequence++
	data["type"] = eventType
	data["sequence_number"] = t.sequence
	raw, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, werr := fmt.Fprintf(t.w, "event: %s\ndata: %s\n\n", eventType, raw); werr != nil {
		return werr
	}
	if t.fl != nil {
		t.fl.Flush()
	}
	return nil
}

// ensureCreated 惰性发出 response.created（首个有效数据帧时）——空流/纯注释流
// 不发 created，直接走 response.failed，避免给客户端一个"创建后无响应"的僵尸对象。
func (t *responsesStreamTranslator) ensureCreated() error {
	if t.created {
		return nil
	}
	t.created = true
	return t.emit("response.created", map[string]any{
		"response": map[string]any{
			"id":         t.responseID,
			"object":     "response",
			"created_at": t.createdAt,
			"status":     "in_progress",
			"model":      t.model,
			"output":     []any{},
		},
	})
}

// nextToolIndex 分配缺 index 的 tool_call 补位序号：跳过既有 index（合规流的
// index 是 0..N-1，补位不能覆盖）。与 sse.go Aggregate 同策略。
func (t *responsesStreamTranslator) nextToolIndex() int {
	for {
		idx := t.toolSeq
		t.toolSeq++
		if _, used := t.toolCalls[idx]; !used {
			return idx
		}
	}
}

// run 主循环：读上游 SSE 行流，逐帧翻译写出。返回错误供观测（空流哨兵 → 502）。
func (t *responsesStreamTranslator) run(r io.Reader) error {
	h := t.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")

	// response.created 立即发出（P1-3）：不再等首个有效帧——空流/错误帧场景客户端
	// 也先拿到 response 对象再收 failed，与官方「created 恒为首个事件」一致。
	if err := t.ensureCreated(); err != nil {
		return err
	}
	br := bufio.NewReaderSize(r, 64*1024)
	for {
		line, err := br.ReadString('\n')
		trimmed := strings.TrimRight(line, "\r\n")
		if strings.HasPrefix(trimmed, "data: ") {
			payload := strings.TrimPrefix(trimmed, "data: ")
			if payload == "[DONE]" {
				// 上游显式结束：DONE 后的数据一律忽略（与 StreamHint 同纪律）。
				t.sawUpstreamDone = true
				break
			}
			if werr := t.handleFrame(payload); werr != nil {
				return werr
			}
			if t.failed {
				// P0-1：error 帧已发 response.failed，终止消费——后续帧若继续翻译
				// 会在 failed 之后产出 item/delta 事件，Codex 按序解析直接报错。
				return nil
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
	}
	if t.validFrames == 0 {
		// 空流：HTTP 200 头已发出（SSE），只能以 response.failed 收尾——
		// 语义与 chat 路径的空流 error 帧 + 502 观测一致。
		t.failed = true
		if werr := t.emit("response.failed", map[string]any{
			"response": map[string]any{
				"id":     t.responseID,
				"object": "response",
				"status": "failed",
				"error":  map[string]any{"message": "empty upstream stream", "type": "upstream_error", "code": "upstream_parse"},
			},
		}); werr != nil {
			return werr
		}
		return errResponsesEmptyStream
	}
	return t.finish()
}

// unwindNestedErrorMessage 原地展开双重编码的 error.message：上游常把业务错误
// 信封 JSON 序列化后塞进 message（如 "{\"code\":6004,\"msg\":\"...\"}"），
// Codex 按 error.code 字符串分类——把内层 code 提升到 error.code、内层 msg 覆盖
// error.message（内层信封才是可读文案；外层原文只是 JSON 壳）。message 不可解析
// 为对象时原样保留（非双重编码形态零改动）。
func unwindNestedErrorMessage(errObj map[string]any) {
	msg, _ := errObj["message"].(string)
	trimmed := strings.TrimSpace(msg)
	if trimmed == "" || trimmed[0] != '{' {
		return
	}
	var inner map[string]any
	if json.Unmarshal([]byte(trimmed), &inner) != nil {
		return
	}
	if _, has := errObj["code"]; !has {
		if code, ok := inner["code"]; ok {
			errObj["code"] = code
		}
	}
	if im, ok := inner["msg"].(string); ok && im != "" {
		errObj["message"] = im
	} else if im, ok := inner["message"].(string); ok && im != "" {
		errObj["message"] = im
	}
}

// handleFrame 处理一帧上游 chat SSE 数据。
func (t *responsesStreamTranslator) handleFrame(payload string) error {
	var chunk map[string]any
	if json.Unmarshal([]byte(payload), &chunk) != nil {
		return nil // 解析失败静默跳过（与 Aggregate 同口径）
	}
	t.validFrames++
	// 上游错误帧（error-passthrough 形态，顶层 error 字段）：翻译为 response.failed
	// 终止——Responses 客户端没有「流中途 error 帧」概念，必须转译为失败事件。
	if e, has := chunk["error"]; has {
		t.failed = true
		errObj, ok := e.(map[string]any)
		if !ok {
			errObj = map[string]any{"message": fmt.Sprintf("%v", e)}
		}
		// 双重编码提升（P1-5）：上游 error.message 常是内层 JSON 字符串
		// （如 "{\"code\":6004,...}"），Codex 从 error.code 找业务码——把内层
		// code/msg 提升到外层，message 保底换内层 msg（更短可读），原文不丢
		// （原文挪到 inner 原 message 位置已失去意义，直接覆盖）。
		unwindNestedErrorMessage(errObj)
		if err := t.ensureCreated(); err != nil {
			return err
		}
		return t.emit("response.failed", map[string]any{
			"response": map[string]any{
				"id":     t.responseID,
				"object": "response",
				"status": "failed",
				"error":  errObj,
			},
		})
	}
	if u, ok := chunk["usage"].(map[string]any); ok {
		t.usage = u // 末帧 usage 权威（前面的帧一般无 usage）
	}
	chs, _ := chunk["choices"].([]any)
	for _, ci := range chs {
		c, _ := ci.(map[string]any)
		if c == nil {
			continue
		}
		if fr, ok := c["finish_reason"].(string); ok && fr != "" {
			t.finishReason = fr
		}
		// deltaConsumedContent：本帧 delta 已消费过 content/reasoning 的标记——
		// 与 Aggregate 的 gotAnyContent latch（sse.go，PR #134）同口径：部分上游
		// 一帧同时带 delta 与整条 message，两路都并入会让正文翻倍。
		deltaConsumedContent := false
		if delta, ok := c["delta"].(map[string]any); ok {
			if rc, _ := delta["reasoning_content"].(string); rc != "" {
				t.appendReasoning(rc)
				deltaConsumedContent = true
			}
			if txt, _ := delta["content"].(string); txt != "" {
				if err := t.appendText(txt); err != nil {
					return err
				}
				deltaConsumedContent = true
			}
			if tcs, ok := delta["tool_calls"].([]any); ok {
				if err := t.appendToolCalls(tcs); err != nil {
					return err
				}
			}
		}
		// 非 delta 的整条 message（部分上游形态）：整条并入（content 一次发为
		// 单个 delta；reasoning 进缓冲；tool_calls 走同一合并路径）。同帧 delta 已
		// 取过正文/reasoning 时跳过这两路（tool_calls 不受 latch 限制——合并幂等）。
		if msg, ok := c["message"].(map[string]any); ok {
			if !deltaConsumedContent {
				if rc, _ := msg["reasoning_content"].(string); rc != "" {
					t.appendReasoning(rc)
				}
				if txt, _ := msg["content"].(string); txt != "" {
					if err := t.appendText(txt); err != nil {
						return err
					}
				}
			}
			if tcs, ok := msg["tool_calls"].([]any); ok {
				if err := t.appendToolCalls(tcs); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// appendReasoning 缓冲思维链片段（64KB 封顶，静默截断——见 reasoningCap 注释）。
// 按剩余额度裁剪而非整帧丢弃：单帧超限（上游一个 delta 就带 >64KB）同样封顶生效，
// 不给异常上游任何绕过窗口。
func (t *responsesStreamTranslator) appendReasoning(txt string) {
	if txt == "" || t.reasoningFlushed {
		return
	}
	remain := reasoningCap - t.reasoningBuf.Len()
	if remain <= 0 {
		return
	}
	if len(txt) > remain {
		txt = txt[:remain]
	}
	t.reasoningBuf.WriteString(txt)
}

// flushReasoning 一次性收口思维链：added + reasoning_summary_part.added +
// reasoning_summary_text.done（完整文本，带 summary_index）+ item.done。
// 触发点：首个正文/工具分片到达（保住 reasoning 的 0 号 output_index 语义）或流结束。
// 事件族选 summary 形态的原因：实测 Codex 只消费 response.reasoning_summary_text.*
// （按 summary_index 定位 summary part），reasoning_text.delta/done 不进 UI——
// 保持一次性收口方案不变，只换事件形态让思维链真正可见。
func (t *responsesStreamTranslator) flushReasoning() error {
	if t.reasoningFlushed || t.reasoningBuf.Len() == 0 {
		return nil
	}
	t.reasoningFlushed = true
	t.reasoningItemID = "rs_" + session.NewMessageID()
	t.reasoningOutIdx = t.nextOutputIndex
	t.nextOutputIndex++
	full := t.reasoningBuf.String()
	item := map[string]any{"type": "reasoning", "id": t.reasoningItemID, "summary": []any{}}
	if err := t.emit("response.output_item.added", map[string]any{
		"output_index": t.reasoningOutIdx, "item": item,
	}); err != nil {
		return err
	}
	if err := t.emit("response.reasoning_summary_part.added", map[string]any{
		"item_id": t.reasoningItemID, "output_index": t.reasoningOutIdx, "summary_index": 0,
		"part": map[string]any{"type": "summary_text", "text": ""},
	}); err != nil {
		return err
	}
	if err := t.emit("response.reasoning_summary_text.done", map[string]any{
		"item_id": t.reasoningItemID, "output_index": t.reasoningOutIdx, "summary_index": 0,
		"text": full,
	}); err != nil {
		return err
	}
	return t.emit("response.output_item.done", map[string]any{
		"output_index": t.reasoningOutIdx,
		"item": map[string]any{
			"type": "reasoning", "id": t.reasoningItemID,
			"summary": []any{map[string]any{"type": "summary_text", "text": full}},
		},
	})
}

// appendText 处理正文增量：首次触发 message item 三连（added + part.added），
// 每片发 output_text.delta。
func (t *responsesStreamTranslator) appendText(txt string) error {
	if txt == "" {
		return nil
	}
	if err := t.ensureCreated(); err != nil {
		return err
	}
	if err := t.flushReasoning(); err != nil {
		return err
	}
	if !t.textStarted {
		t.textStarted = true
		t.textItemID = "msg_" + session.NewMessageID()
		t.textOutIdx = t.nextOutputIndex
		t.nextOutputIndex++
		if err := t.emit("response.output_item.added", map[string]any{
			"output_index": t.textOutIdx,
			"item": map[string]any{
				"type": "message", "id": t.textItemID, "role": "assistant",
				"status": "in_progress", "content": []any{},
			},
		}); err != nil {
			return err
		}
		if err := t.emit("response.content_part.added", map[string]any{
			"item_id": t.textItemID, "output_index": t.textOutIdx, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}},
		}); err != nil {
			return err
		}
	}
	// 正文封顶同 reasoning（P1-2）：按剩余额度裁剪；超限后静默丢弃后续增量
	// （不再发 delta——发出去的与 done 的全文必须一致，delta 是已缓冲前缀）。
	remain := textCap - t.textBuf.Len()
	if remain <= 0 {
		return nil
	}
	truncated := false
	if len(txt) > remain {
		txt = txt[:remain]
		truncated = true
	}
	t.textBuf.WriteString(txt)
	if truncated {
		t.textTruncated = true
	}
	return t.emit("response.output_text.delta", map[string]any{
		"item_id": t.textItemID, "output_index": t.textOutIdx, "content_index": 0, "delta": txt,
	})
}

// appendToolCalls 处理一帧内的 tool_calls 数组：按 index 归位（缺 index 按 id、
// 再退最近调用——与 sse.go mergeToolCallsChunk 同策略）。流中**只缓冲不发射**：
// added/delta/done 三族事件全部延迟到 finish() 统一收口——截断判定（残缺参数
// 剔除）依赖 finish_reason 与 [DONE] 两个流尾信号，若流中先发了 added/delta，
// 截断调用已无法从事件流里撤回，Codex 会拿到残缺 arguments（P1-1）。
// 代价是工具调用首字节的到流延迟（与正文 delta 分帧同量级，可接受）。
func (t *responsesStreamTranslator) appendToolCalls(tcs []any) error {
	if err := t.ensureCreated(); err != nil {
		return err
	}
	if err := t.flushReasoning(); err != nil {
		return err
	}
	for _, tc := range tcs {
		call, ok := tc.(map[string]any)
		if !ok {
			continue
		}
		idx := -1
		if v, ok := call["index"].(float64); ok {
			idx = int(v)
		} else if cid, _ := call["id"].(string); cid != "" {
			if mid, seen := t.idIndex[cid]; seen {
				idx = mid // 该 id 已归位：跨帧延续既有调用
			} else {
				idx = t.nextToolIndex()
			}
		} else if len(t.toolOrder) > 0 {
			idx = t.toolOrder[len(t.toolOrder)-1] // 无 id 碎片：延续最近调用
		} else {
			idx = t.nextToolIndex()
		}
		state, seen := t.toolCalls[idx]
		if !seen {
			state = &responsesToolCallState{
				outIdx: t.nextOutputIndex,
				itemID: "fc_" + session.NewMessageID(),
			}
			t.nextOutputIndex++
			t.toolCalls[idx] = state
			t.toolOrder = append(t.toolOrder, idx)
		}
		if cid, _ := call["id"].(string); cid != "" {
			t.idIndex[cid] = idx
			if state.callID == "" {
				state.callID = cid
			}
		}
		df, _ := call["function"].(map[string]any)
		if nm, _ := df["name"].(string); nm != "" && state.name == "" {
			state.name = nm
		}
		// 只缓冲：name 存 state.name、arguments 片段拼接进 state.args，
		// added/delta/done 事件族在 finish() 的 emitToolCallItems 统一发射。
		if frag, _ := df["arguments"].(string); frag != "" {
			state.args.WriteString(frag)
		}
	}
	return nil
}

// finish 流结束收口：reasoning → 文本三连 done → 各 tool call done →
// response.completed。finish_reason=="length" 时 status=incomplete（带
// incomplete_details），对齐官方截断语义。
func (t *responsesStreamTranslator) finish() error {
	if t.failed {
		return nil // 已失败终止（error 帧路径），不再补 completed
	}
	if err := t.flushReasoning(); err != nil {
		return err
	}
	if t.textStarted {
		full := t.textBuf.String()
		if err := t.emit("response.output_text.done", map[string]any{
			"item_id": t.textItemID, "output_index": t.textOutIdx, "content_index": 0, "text": full,
		}); err != nil {
			return err
		}
		if err := t.emit("response.content_part.done", map[string]any{
			"item_id": t.textItemID, "output_index": t.textOutIdx, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": full, "annotations": []any{}},
		}); err != nil {
			return err
		}
		if err := t.emit("response.output_item.done", map[string]any{
			"output_index": t.textOutIdx,
			"item": map[string]any{
				"type": "message", "id": t.textItemID, "role": "assistant",
				"status":  "completed",
				"content": []any{map[string]any{"type": "output_text", "text": full, "annotations": []any{}}},
			},
		}); err != nil {
			return err
		}
	}
	// 截断判定（P1-1，与 chat 非流式 dropTruncatedToolCalls 同口径单一来源）：
	// 非空但解析失败的参数是残缺 JSON——下发会让 Codex 解析报错卡死会话。
	// 处置照抄 chat 语义：截断的调用整体剔除（added/delta/done 都不发，事件流里
	// 消失——工具事件已延迟到本函数发射，剔除天然完整），completed.status 收敛
	// incomplete；非空完整参数（含无参空串）零改动。
	truncatedByLength := t.finishReason == "length" || !t.sawUpstreamDone
	for _, idx := range t.toolOrder {
		state := t.toolCalls[idx]
		full := state.args.String()
		if truncatedByLength && upstream.IsTruncatedArguments(full) {
			t.droppedTruncatedTool = true
			log.Printf("WARN: [server] responses stream: drop truncated tool call call_id=%s name=%q", state.callID, state.name)
			continue
		}
		if err := t.emitToolCallItems(state, full); err != nil {
			return err
		}
	}
	resp := map[string]any{
		"id":         t.responseID,
		"object":     "response",
		"created_at": t.createdAt,
		"status":     "completed",
		"model":      t.model,
		"output":     t.outputItems(),
	}
	if t.finishReason == "length" || t.droppedTruncatedTool || t.textTruncated {
		// 截断语义（含 P1-1 剔除命中与 P1-2 正文封顶命中）：报 incomplete +
		// max_output_tokens，不再报 completed——客户端据此知道本轮输出不完整。
		resp["status"] = "incomplete"
		resp["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
	}
	if t.usage != nil {
		// P0-2：Codex 的 ResponseCompletedUsage.total_tokens 是必填字段，缺失即整轮
		// 解析失败。上游末帧确实会漏 total（chat 非流式有 ensureUsageTotal 单点兜底，
		// 这里复用同一导出实现——input+output 合成，缺一不编造）。
		resp["usage"] = responsesUsage(upstream.EnsureUsageTotal(t.usage))
	} else {
		resp["usage"] = nil // 缺失置 null（不编造零值）
	}
	return t.emit("response.completed", map[string]any{"response": resp})
}

// emitToolCallItems 发射单个 tool call 的完整事件三连（added → arguments.delta →
// arguments.done → item.done）。流中缓冲的 arguments 以一次 delta 全量发出：Codex
// 按 delta 拼接还原，事件形态与官方一致；增量粒度损失不影响正确性（工具事件本就
// 延迟到收口，见 appendToolCalls 注释）。
func (t *responsesStreamTranslator) emitToolCallItems(state *responsesToolCallState, full string) error {
	// namespace 回译（见 toolMap 字段注释）：拼接名命中映射表才拆回
	// (real_name, namespace)；未命中保持拼接名原样、不填 namespace。
	itemName, ns, _ := t.responsesToolCallIdentity(state.name)
	item := func(status string, args string) map[string]any {
		m := map[string]any{
			"type": "function_call", "id": state.itemID,
			"call_id": state.callID, "name": itemName,
			"arguments": args, "status": status,
		}
		if ns != "" {
			m["namespace"] = ns
		}
		return m
	}
	if err := t.emit("response.output_item.added", map[string]any{
		"output_index": state.outIdx,
		"item":         item("in_progress", ""),
	}); err != nil {
		return err
	}
	state.added = true
	if full != "" {
		if err := t.emit("response.function_call_arguments.delta", map[string]any{
			"item_id": state.itemID, "output_index": state.outIdx, "delta": full,
		}); err != nil {
			return err
		}
	}
	if err := t.emit("response.function_call_arguments.done", map[string]any{
		"item_id": state.itemID, "output_index": state.outIdx, "arguments": full,
	}); err != nil {
		return err
	}
	return t.emit("response.output_item.done", map[string]any{
		"output_index": state.outIdx,
		"item":         item("completed", full),
	})
}

// outputItems 按输出序号归集全量 items（completed 事件里的 response.output）。
// 截断剔除的 tool call 不出现在 output（与事件流收口行为一致）；reasoning 的
// summary part 携带完整文本（与 flushReasoning 收口事件同形态）。
func (t *responsesStreamTranslator) outputItems() []any {
	type entry struct {
		idx  int
		item map[string]any
	}
	var entries []entry
	if t.reasoningFlushed && t.reasoningBuf.Len() > 0 {
		full := t.reasoningBuf.String()
		entries = append(entries, entry{t.reasoningOutIdx, map[string]any{
			"type": "reasoning", "id": t.reasoningItemID,
			"summary": []any{map[string]any{"type": "summary_text", "text": full}},
		}})
	}
	if t.textStarted {
		full := t.textBuf.String()
		entries = append(entries, entry{t.textOutIdx, map[string]any{
			"type": "message", "id": t.textItemID, "role": "assistant", "status": "completed",
			"content": []any{map[string]any{"type": "output_text", "text": full, "annotations": []any{}}},
		}})
	}
	truncatedByLength := t.finishReason == "length" || !t.sawUpstreamDone
	for _, idx := range t.toolOrder {
		state := t.toolCalls[idx]
		full := state.args.String()
		if truncatedByLength && upstream.IsTruncatedArguments(full) {
			continue // 已在 finish() 记 WARN 与 incomplete 标记，这里只静默排除
		}
		// namespace 回译与 emitToolCallItems 同一函数（responsesToolCallIdentity），
		// completed 事件 output 里的 item 与事件流逐 item done 保持一致。
		itemName, ns, _ := t.responsesToolCallIdentity(state.name)
		fc := map[string]any{
			"type": "function_call", "id": state.itemID, "call_id": state.callID,
			"name": itemName, "arguments": full, "status": "completed",
		}
		if ns != "" {
			fc["namespace"] = ns
		}
		entries = append(entries, entry{state.outIdx, fc})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].idx < entries[j].idx })
	out := make([]any, 0, len(entries))
	for _, e := range entries {
		out = append(out, e.item)
	}
	return out
}

// responsesStreamSuccess responses 流式成功路径（pipeSink 回调）：chatStatsReader
// 包裹上游流做观测（TTFB/tokens/credit——逐行扫只取统计、字节原样转发），翻译器
// 消费同一 reader 写出 Responses 事件流。返回值与 chat sink 同语义：是否成功收尾
// （false = 空流，观测收敛 502）。
func (s responsesPipeSink) onStreamSuccess(w http.ResponseWriter, h *Handler, acct *auth.Auth, rc io.ReadCloser, st *chatStat, bareModel string, _ func() upstream.HintContext) bool {
	// 流式：翻译结束后立即关闭上游 body，避免 defer 在轮转场景下堆积 fd。
	st.status = http.StatusOK
	stats := newChatStatsReaderSince(rc, st.start)
	tr := newResponsesStreamTranslator(w, bareModel)
	// namespace 工具映射表下沉（ResponsesToChat 展开时生成）：发射 function_call
	// item 时按表回译 name/namespace（见翻译器 toolMap 字段注释）。
	tr.toolMap = s.toolMap
	sErr := tr.run(stats)
	rc.Close()
	if errors.Is(sErr, errResponsesEmptyStream) {
		// 上游 200 但空流（0 有效帧）：翻译器已发 response.failed 收尾（头已 200），
		// 观测收敛 502——与非流式 Aggregate 空流→502、chat 流空流观测同语义。
		st.status = http.StatusBadGateway
		log.Printf("WARN: [server] responses stream acct=%s model=%s: empty upstream stream (200+0 frames)", logfmt.Label(acct.UID, acct.Nickname), bareModel)
	} else if sErr != nil {
		// 客户端断连/写失败：人已走，无观测意义，仅保留 200（与 chat 路径口径一致）。
		log.Printf("WARN: [server] responses stream write aborted acct=%s model=%s: %v", logfmt.Label(acct.UID, acct.Nickname), bareModel, sErr)
	}
	st.ttfb = stats.TTFB()
	// usage 缺失时保留 -1 哨兵（观测缺失 → 显示 "-"），与 chat 同口径。
	toks, hasUsage := stats.Tokens()
	if hasUsage {
		st.toks = toks
	}
	// 成本账本：末帧 usage 带 credit 时记实测单价（与 chat 完全同语义）。
	if credit, ok := stats.Credit(); ok {
		h.cfg.Pool.NoteModelCost(acct.UID, bareModel, credit, stats.TotalTokens())
	} else if hasUsage {
		log.Printf("WARN: [server] responses stream usage without credit acct=%s model=%s (no cost observation)", logfmt.Label(acct.UID, acct.Nickname), bareModel)
	}
	return sErr == nil
}
