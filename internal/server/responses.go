// responses.go OpenAI Responses API 协议翻译层（POST /v1/responses）。
//
// 背景：OpenAI Codex CLI 已硬移除 chat wire_api，只发 Responses 协议；本文件把
// Responses 请求翻译为标准 chat/completions body，走与 chat 完全相同的共享轮转
// 管线（pipeline.go runChatPipeline——粘性、选号、熔断、成本账本、错误策略同语义），
// 出站前仍经 upstream payload.go 统一预处理（强制 stream、tool 配对、thinking 注入）。
// 响应侧由 responses_stream.go 的流式翻译器 / 本文件的聚合翻译回 Responses 事件流/对象。
package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"workbuddy2api/internal/auth"
	"workbuddy2api/internal/session"
	"workbuddy2api/internal/upstream"
)

// knownServerToolOnce 按类型各自一次的服务端托管工具剔除降噪（web_search /
// tool_search 是 Codex 每请求的标准组成，逐请求日志是纯噪音；不同类型首次出现
// 各记一条 INFO，便于运维知道具体丢了哪些能力）。map 键只增不改，锁保护读写。
var knownServerToolOnce = struct {
	sync.Mutex
	seen map[string]bool
}{seen: map[string]bool{}}

// ResponsesToChat 把 Responses 请求体翻译为标准 chat/completions 请求体（兼容入口：
// 不关心 namespace 工具映射的调用方/既有测试用；handler 走 responsesToChatWithToolMap
// 额外取映射表做响应侧回译）。
func ResponsesToChat(body []byte) ([]byte, error) {
	chatBody, _, err := responsesToChatWithToolMap(body)
	return chatBody, err
}

// responsesToolIdentity namespace 工具展开的映射值：Namespace 是 Responses namespace
// 工具的 name（如 multi_agent_v1），Name 是子工具真实名（如 spawn_agent）。映射表
// key 一律为 joinNamespacedToolName 的拼接名（即模型可见的 chat 工具名）。
type responsesToolIdentity struct {
	Namespace string
	Name      string
}

// joinNamespacedToolName namespace 工具名的唯一拼接规则：namespace 与子工具名
// **直接拼接、无分隔符**——依据 Codex 自身 ToolName Display 规则（codex-rs/protocol/
// src/tool_name.rs：write!(f, "{namespace}{}", self.name)，形如 multi_agent_v1spawn_agent，
// 看似无分隔但这是 Codex 的解析口径）。请求侧展开、history 回传拼接、响应侧回译
// 查表三处共用本函数，保证模型看到的工具名与 Codex 期望的 function_call
// (name+namespace) 形态严格互译，不出现第二种拼法。
func joinNamespacedToolName(namespace, name string) string {
	return namespace + name
}

// responsesToChatWithToolMap ResponsesToChat 的完整实现，额外返回 namespace 工具
// 映射表（拼接名 → 真实身份，供流式/非流式响应回译 function_call 的 name/namespace）。
// 只做「协议字段映射」，不做任何业务预处理（那是 upstream payload.go 的职责）：
//   - instructions → 最前一条 system 消息
//   - input：string → 单条 user 消息；数组 → 逐项映射（message/function_call/
//     function_call_output/reasoning）
//   - tools 扁平形态 → 嵌套形态（上游只认 {"type":"function","function":{...}}）；
//     namespace 类型（Codex multi_agent 委派工具包）按拼接规则逐子工具展开为
//     chat function——chat 协议没有命名空间概念，整体剔除会让上游模型看不到
//     spawn_agent 等委派工具，Codex 的 subagent 功能即整体失效
//   - reasoning.effort → 顶层 reasoning_effort；max_output_tokens → max_tokens
//   - metadata.conversation / conversation（Codex 形态）→ metadata.conversation_id
//     （会话粘性键，session.ExtractKey 命中）
//
// 返回翻译后的 body（必定含 messages，可通过 session.TurnKey 派生轮级键；instructions
// 与 input 合计无可用内容时报错，不产出空 messages body）与映射表；请求不含
// namespace 工具时映射表为 nil（回译侧全走原样透传的零回归路径）。
func responsesToChatWithToolMap(body []byte) ([]byte, map[string]responsesToolIdentity, error) {
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, nil, fmt.Errorf("invalid JSON body: %w", err)
	}
	chat := map[string]any{}
	toolMap := map[string]responsesToolIdentity{}

	// model 原样保留（含 realm 前缀——"[realm:]model" 是网关侧路由协议，
	// 由 chatCompletions 同款 resolveModel 剥离，选号/粘性/账本共用该语义）。
	if m, ok := req["model"].(string); ok && m != "" {
		chat["model"] = m
	}

	// messages 组装：instructions 提为最前 system；input 按形态展开。
	msgs := make([]any, 0, 8)
	if ins, ok := req["instructions"].(string); ok && ins != "" {
		msgs = append(msgs, map[string]any{"role": "system", "content": ins})
	}
	inputMsgs, err := responsesInputToMessages(req["input"], toolMap)
	if err != nil {
		return nil, nil, err
	}
	msgs = append(msgs, inputMsgs...)
	if len(msgs) == 0 {
		// P1-6/L2：instructions 与 input 合计无可用消息（input 为空串/全跳空且无
		// instructions）→ 空 messages 出站会让上游 400 11101（网关罚号轮空，客户端
		// 还得等一轮超时）。请求本身无可用内容，翻译层直接报错终止不进轮转。
		return nil, nil, fmt.Errorf("input contained no usable messages")
	}
	chat["messages"] = msgs

	// tools：Responses 扁平形态 → chat 嵌套形态；strict 字段删除（上游不认，
	// 透传徒增 11101 参数错误风险）。已是嵌套形态（chat 形态）则原样透传。
	// 非 function 形态：custom（freeform，Codex apply_patch 用）映射为单字符串参数的
	// function（上游只认 function）；namespace（Codex multi_agent 委派工具包，见下）
	// 逐子工具展开；tool_search/web_search 等服务端工具记 WARN 后剔除（上游无对应
	// 语义，透传必 400）。
	if tools, ok := req["tools"].([]any); ok {
		out := make([]any, 0, len(tools))
		for _, t := range tools {
			tm, ok := t.(map[string]any)
			if !ok {
				continue // 非对象条目丢弃（畸形工具定义不值得整请求失败）
			}
			typ, _ := tm["type"].(string)
			if typ == "custom" {
				// freeform 文本工具 → function（input 单字符串参数）：模型侧调用形态
				// 变成 {"input": "..."}，与 responsesCustomToolCallItem 的 arguments
				// 直通形成闭环（Codex 侧仍按 custom 工具解析补丁文本）。
				name, _ := tm["name"].(string)
				if name == "" {
					continue
				}
				out = append(out, map[string]any{"type": "function", "function": map[string]any{
					"name":        name,
					"description": tm["description"],
					"parameters": map[string]any{
						"type":       "object",
						"properties": map[string]any{"input": map[string]any{"type": "string"}},
					},
				}})
				continue
			}
			if typ == "namespace" {
				// namespace 类型（Codex multi_agent_v1 委派工具包）：chat 协议没有
				// 命名空间概念，剔除会让上游模型看不到 spawn_agent 等委派工具
				// （auto-delegate 类 subagent 功能整体失效）。展开规则照 Codex 自身
				// ToolName Display 口径（tool_name.rs：namespace+name 直接拼接，
				// 无分隔符）——每个子工具成为一条 chat function，拼接名进 toolMap
				// 供响应侧回译（模型 tool_call 回传时按拼接名反查 name/namespace）。
				nsName, _ := tm["name"].(string)
				if nsName == "" {
					log.Printf("WARN: [server] responses tools: namespace tool without name dropped")
					continue
				}
				nsDesc, _ := tm["description"].(string)
				subTools, _ := tm["tools"].([]any)
				for _, st := range subTools {
					stm, ok := st.(map[string]any)
					if !ok {
						continue
					}
					subFn := responsesNamespaceSubTool(stm, nsName, nsDesc, toolMap)
					if subFn != nil {
						out = append(out, subFn)
					}
				}
				continue
			}
			if typ == "function" {
				if _, nested := tm["function"].(map[string]any); !nested {
					fn := map[string]any{}
					for _, k := range []string{"name", "description", "parameters"} {
						if v, ok := tm[k]; ok {
							fn[k] = v
						}
					}
					out = append(out, map[string]any{"type": "function", "function": fn})
					continue
				}
				out = append(out, tm)
				continue
			}
			// 其余类型分流降噪：web_search/tool_search 是 Codex 每请求的标准组成
			// （OpenAI 服务端托管工具，chat 上游无法模拟，剔除是既定行为），逐请求
			// WARN 会淹没日志——按类型首次出现各记一条 INFO；未知类型（可能是
			// 新协议字段）仍逐请求 WARN，丢失信息需要客户端可见。
			if typ == "web_search" || typ == "tool_search" {
				knownServerToolOnce.Lock()
				first := !knownServerToolOnce.seen[typ]
				knownServerToolOnce.seen[typ] = true
				knownServerToolOnce.Unlock()
				if first {
					log.Printf("INFO: [server] responses tools: server-executed type %q dropped (chat upstream cannot simulate); further drops of this type are silent", typ)
				}
				continue
			}
			log.Printf("WARN: [server] responses tools: unsupported type %q (name=%v) dropped", typ, tm["name"])
		}
		if len(out) > 0 {
			chat["tools"] = out
		}
	}

	// tool_choice 透传（upstream payload.go normalizeToolChoice 统一归一化——
	// 上游只认 string，对象形式由其翻译，翻译层不重复实现）。
	if tc, ok := req["tool_choice"]; ok {
		chat["tool_choice"] = tc
	}

	// reasoning.effort → reasoning_effort（Responses 对象形态 → chat 顶层标量；
	// 后续 upstream normalizeReasoningEffort 按模型档位降级）。
	if r, ok := req["reasoning"].(map[string]any); ok {
		if e, ok := r["effort"].(string); ok && e != "" {
			chat["reasoning_effort"] = e
		}
	}

	// max_output_tokens → max_tokens（显式 max_tokens 优先不覆盖——Responses
	// 规范里没有 max_tokens，此守卫纯防御手构造请求）。数值防御与 upstream
	// translateMaxCompletionTokens 同口径：null/0/负数语义是「未设置」，非数值
	// 原样透传由上游报 11101——只有正整数才翻译。
	if v, ok := req["max_output_tokens"]; ok {
		if _, exists := chat["max_tokens"]; !exists {
			switch n := v.(type) {
			case float64:
				if n > 0 && n == float64(int64(n)) {
					chat["max_tokens"] = int64(n)
				}
			case int64:
				if n > 0 {
					chat["max_tokens"] = n
				}
			case int:
				if n > 0 {
					chat["max_tokens"] = int64(n)
				}
			}
		}
	}

	// 采样参数透传。
	for _, k := range []string{"temperature", "top_p"} {
		if v, ok := req[k]; ok {
			chat[k] = v
		}
	}

	// stream 保留（Codex 恒 true；false 也支持——非流式走 Aggregate 聚合翻译）。
	if v, ok := req["stream"]; ok {
		chat["stream"] = v
	}

	// 会话粘性键映射：Codex 发 metadata.conversation 或顶层 conversation（对象或
	// 字符串形态均见过），统一映射为 chat body 的 metadata.conversation_id，让
	// session.ExtractKey 命中粘性路由（同对话钉同号，上游 prompt cache 不碎）。
	// 对象形态取其 id 字段；字符串形态直接用。
	convID := responsesConversationID(req)
	if convID != "" {
		chat["metadata"] = map[string]any{"conversation_id": convID}
	}

	out, err := json.Marshal(chat)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal translated chat body: %w", err)
	}
	// 无 namespace 工具时映射表保持空（回 nil 而非空 map——回译侧以 nil 判定
	// 零回归路径，避免每请求空表查询与分配）。
	if len(toolMap) == 0 {
		toolMap = nil
	}
	return out, toolMap, nil
}

// responsesNamespaceSubTool 把 namespace 的单个子工具展开为 chat 嵌套 function 工具。
// 子工具是 ResponsesApiNamespaceTool 枚举（type: "function" | "custom"，custom 为
// Codex freeform apply_patch 类）：
//   - 拼接名 joinNamespacedToolName(ns, sub)（与 Codex tool_name.rs Display 严格一致，
//     模型回传 tool_call 时按同名反查映射表回译）；
//   - description：namespace 有非空描述时附加前缀（"[namespace ns: desc] " + 原文），
//     帮模型区分同批次里来自不同命名空间的同名工具；
//   - parameters：function 子工具透传；custom 子工具缺失时兜底单字符串 input 参数
//     （freeform 工具参数是自由文本，模型侧调用形态 {"input": "..."}）。
//
// 子工具名与 namespace 同名等重名冲突/缺名的畸形条目跳过并 WARN（防御：拼接名
// 撞车会污染 toolMap 与历史工具名一致性）。
func responsesNamespaceSubTool(stm map[string]any, nsName, nsDesc string, toolMap map[string]responsesToolIdentity) map[string]any {
	subTyp, _ := stm["type"].(string)
	subName, _ := stm["name"].(string)
	if subName == "" {
		log.Printf("WARN: [server] responses tools: namespace %q sub-tool without name dropped", nsName)
		return nil
	}
	if subName == nsName {
		// 拼接名歧义防御：ns+"ns" 与子工具名撞车不会发生在合法 Codex 请求里，
		// 但拼接规则下无法与真实同名子工具区分，宁缺毋滥。
		log.Printf("WARN: [server] responses tools: namespace %q sub-tool name collides with namespace, dropped", nsName)
		return nil
	}
	var params any
	switch subTyp {
	case "function":
		params = stm["parameters"]
	case "custom":
		// freeform：parameters 缺失（合法形态）时兜底单字符串 input。
		if p, ok := stm["parameters"].(map[string]any); ok {
			params = p
		} else {
			params = map[string]any{
				"type":       "object",
				"properties": map[string]any{"input": map[string]any{"type": "string"}},
			}
		}
	default:
		// namespace 内的未知子工具类型（Codex 枚举只承诺 function|custom）：
		// 记 WARN 跳过，不影响其余子工具展开。
		log.Printf("WARN: [server] responses tools: namespace %q unsupported sub-tool type %q (name=%v) dropped", nsName, subTyp, stm["name"])
		return nil
	}
	joined := joinNamespacedToolName(nsName, subName)
	desc, _ := stm["description"].(string)
	if nsDesc != "" {
		desc = fmt.Sprintf("[namespace %s: %s] %s", nsName, nsDesc, desc)
	}
	// 映射表恒等原则：本次请求展开的每个子工具都登记，模型任何一次 tool_call 的
	// 拼接名都能反查到 (namespace, real_name)——响应侧据此回译 function_call item。
	toolMap[joined] = responsesToolIdentity{Namespace: nsName, Name: subName}
	fn := map[string]any{"name": joined, "description": desc, "parameters": params}
	return map[string]any{"type": "function", "function": fn}
}

// responsesConversationID 从 Responses 请求提取会话粘性键：
// metadata.conversation → 顶层 conversation；对象形态取 id 字段，字符串形态直接用。
func responsesConversationID(req map[string]any) string {
	extract := func(v any) string {
		switch t := v.(type) {
		case string:
			return t
		case map[string]any:
			id, _ := t["id"].(string)
			return id
		}
		return ""
	}
	if meta, ok := req["metadata"].(map[string]any); ok {
		if id := extract(meta["conversation"]); id != "" {
			return id
		}
		if id := extract(meta["conversation_id"]); id != "" {
			return id // chat 形态的 metadata.conversation_id 已被部分客户端复用
		}
	}
	return extract(req["conversation"])
}

// responsesInputToMessages 把 Responses 的 input 字段展开为 chat messages。
// string → 单条 user；数组 → 逐项映射：
//   - message：content 为 string 或 parts 数组（input_text/output_text/summary_text
//     拼接 text；input_image → image_url part，维持多模态通路）
//   - function_call → assistant tool_calls（content 置 ""，tool 配对由 payload.go 清理）；
//     带 namespace 字段的历史 function_call/custom_tool_call（Codex 上一轮原样回传）
//     按 joinNamespacedToolName 拼接为 chat 工具名，与请求侧展开的 toolMap 恒等，
//     模型看到的历史工具名与自己上轮发出的拼接名一致
//   - function_call_output → tool 消息（output 为对象时序列化为字符串）
//   - reasoning → 跳过（不发给上游；上游思维链由 thinking.go 注入回填）
//   - 未知 type → 跳过并 WARN（新客户端字段演进时可见，不静默吞）
//
// toolMap 为本请求 namespace 工具映射表（可为 nil——无 namespace 工具时零参与）。
// 它同时承担两项职责：**input 历史回传先行处理时**登记 namespace 工具（客户端可能
// 回传本网关未见过的命名空间工具，如会话跨网关续接——登记保证下一轮模型发起调用
// 时映射表仍能回译）；随后请求 tools 展开以权威源身份覆盖/补全映射（见调用方
// responsesToChatWithToolMap 的组装顺序：input 先于 tools 展开）。
func responsesInputToMessages(input any, toolMap map[string]responsesToolIdentity) ([]any, error) {
	if input == nil {
		return nil, nil
	}
	switch v := input.(type) {
	case string:
		if v == "" {
			return nil, nil
		}
		return []any{map[string]any{"role": "user", "content": v}}, nil
	case []any:
		msgs := make([]any, 0, len(v))
		// pendingAssistant 连续 function_call/custom_tool_call 的合并缓冲：
		// Responses 协议并行工具调用是逐 item 平铺（每 call 一个 function_call
		// item），chat 协议要求并行的 tool_calls 合并在**同一条** assistant 消息里
		// （上游 tool 配对 11148 按消息粒度校验——拆成多条 assistant 各带一个
		// tool_call，上游会判「调了 A 没等结果就调 B」而拒单，Codex 实测复现）。
		// 遇到非 tool_call item（message/function_call_output 等）时冲刷缓冲。
		var pendingAssistant map[string]any
		flushPending := func() {
			if pendingAssistant != nil {
				msgs = append(msgs, pendingAssistant)
				pendingAssistant = nil
			}
		}
		appendToPending := func(call map[string]any) {
			if pendingAssistant == nil {
				pendingAssistant = map[string]any{"role": "assistant", "content": "", "tool_calls": []any{}}
			}
			pendingAssistant["tool_calls"] = append(pendingAssistant["tool_calls"].([]any), call)
		}
		for i, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				log.Printf("WARN: [server] responses input[%d]: non-object item skipped", i)
				continue
			}
			typ, _ := m["type"].(string)
			switch typ {
			case "message":
				flushPending()
				msg := responsesMessageItem(m)
				if msg != nil {
					msgs = append(msgs, msg)
				}
			case "function_call":
				if call := responsesToolCallEntry(m, "arguments", toolMap); call != nil {
					appendToPending(call)
				}
			case "function_call_output":
				flushPending()
				msg := responsesFunctionCallOutputItem(m)
				if msg != nil {
					msgs = append(msgs, msg)
				}
			case "reasoning":
				// 跳过：上游思维链由 thinking.go 注入（thinking.type=enabled），
				// 历史 reasoning 若回传反而与回填机制冲突。不冲刷 pending——
				// Codex 历史里 reasoning 紧邻其发起的 function_call（同轮产物）。
			case "custom_tool_call":
				// P1-6：Codex apply_patch freeform 形态——结构同 function_call，
				// 参数在 input 字段（非 JSON，通常是纯文本补丁）。映射为 function
				// tool_call（input 即 arguments 字符串），上游按普通函数调用回传，
				// Codex 侧自会按 custom 工具解析。带 namespace 的历史帧同 function_call
				// 规则拼接（Codex 的 multi_agent freeform 子工具同样有 namespace 字段）。
				if call := responsesToolCallEntry(m, "input", toolMap); call != nil {
					appendToPending(call)
				}
			case "custom_tool_call_output":
				flushPending()
				msg := responsesCustomToolCallOutputItem(m)
				if msg != nil {
					msgs = append(msgs, msg)
				}
			case "":
				// 无 type 字段：按 message 宽容处理（部分客户端省略 type）。
				flushPending()
				msg := responsesMessageItem(m)
				if msg != nil {
					msgs = append(msgs, msg)
				}
			default:
				// 未知 type 记 WARN 后跳过（不整请求失败——新客户端字段演进时可见
				// 即可）；全数组被跳空时外层统一报错（下方 len 校验）。
				log.Printf("WARN: [server] responses input[%d]: unknown item type %q skipped", i, typ)
			}
		}
		flushPending()
		// 注意：这里**不**做 len(msgs)==0 报错——input 全跳空但 instructions 有值时
		// 合法（system-only 请求），空 messages 的统一判定在**外层**组装完成后做
		// （instructions + input 合计为空才 400，见 responsesToChatWithToolMap）。
		return msgs, nil
	default:
		return nil, fmt.Errorf("invalid input: must be string or array")
	}
}

// responsesMessageItem 映射 {"type":"message",role,content} 为 chat 消息。
// role 缺省按 user 宽容处理（Codex 历史帧恒带 role，此为手构造请求兜底）。
func responsesMessageItem(m map[string]any) map[string]any {
	role, _ := m["role"].(string)
	if role == "" {
		role = "user"
	}
	content := responsesContent(m["content"])
	if content == nil {
		return nil // content 缺失（如纯 reasoning 帧残片）无可发，跳过
	}
	return map[string]any{"role": role, "content": content}
}

// responsesContent 把 Responses content（string 或 parts 数组）翻译为 chat content。
//   - string → 原样
//   - parts：input_text/output_text/summary_text → 拼接 text（string 形态返回）；
//     input_image → {"type":"image_url","image_url":...}（出现任一图片 part 时
//     整体升级为 chat 多模态数组形态）；无法归类的 part 跳过
//   - 其他类型（null 等）→ nil
func responsesContent(v any) any {
	switch t := v.(type) {
	case string:
		return t
	case []any:
		var b strings.Builder
		var parts []any
		hasImage := false
		for _, p := range t {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			pt, _ := pm["type"].(string)
			switch pt {
			case "input_text", "output_text", "summary_text":
				if txt, ok := pm["text"].(string); ok {
					b.WriteString(txt)
				}
			case "input_image":
				// Responses 图片 part 的 URL 字段两种形态都见过：image_url（字符串）
				// 与 image_url（对象 {url}）；chat 形态是 {"type":"image_url",
				// "image_url":"<url>"}，统一归一到字符串。
				switch img := pm["image_url"].(type) {
				case string:
					parts = append(parts, map[string]any{"type": "image_url", "image_url": img})
					hasImage = true
				case map[string]any:
					if u, ok := img["url"].(string); ok && u != "" {
						parts = append(parts, map[string]any{"type": "image_url", "image_url": u})
						hasImage = true
					}
				}
			}
		}
		if hasImage {
			// 多模态：文本 part 排最前（chat 惯例），图片随后。
			if b.Len() > 0 {
				return append([]any{map[string]any{"type": "text", "text": b.String()}}, parts...)
			}
			return parts
		}
		return b.String()
	}
	return nil
}

// responsesToolCallEntry 把 function_call / custom_tool_call item 映射为 chat
// tool_calls 数组条目（{"id","type":"function","function":{name,arguments}}）。
// argField 指定参数来源字段：function_call 用 "arguments"（JSON 串），
// custom_tool_call 用 "input"（自由文本，原样作为 arguments 下发）。
// 历史回传拼接（关键）：item 带非空 namespace 字段且 != "functions" 时，chat 侧
// 工具名取 joinNamespacedToolName(namespace, name)——与请求侧展开规则恒等，模型
// 看到的历史工具名与它上一轮自己发出的拼接名严格一致；"functions" 是 OpenAI
// 无命名空间工具的缺省命名空间（照 Codex 客户端口径），拼接反而会造出不存在的
// 工具名。同时登记进 toolMap（历史工具可能不在本请求 tools 展开集合里）。
// item 上的 name/namespace 不参与 function_call_output / custom_tool_call_output 的
// tool 消息构造（按 call_id 配对，忽略冗余字段）。
// 无函数名的 call 是残片返回 nil（配对无意义）；call_id 缺省生成占位
// （tool_call_id 配对键必须两侧一致，缺 id 的 call 无法配对其 output，
// 此处仍保留结构让 payload.go 的孤儿清理兜底）。
func responsesToolCallEntry(m map[string]any, argField string, toolMap map[string]responsesToolIdentity) map[string]any {
	name, _ := m["name"].(string)
	if name == "" {
		return nil
	}
	if ns, _ := m["namespace"].(string); ns != "" && ns != "functions" {
		realName := name
		name = joinNamespacedToolName(ns, name)
		if toolMap != nil {
			// 只登记尚不存在的条目：请求 tools 展开的映射是权威源（其 description
			// 等更完整），历史回传仅补洞，不覆盖。Name 存真实名（非拼接名）——
			// 回译时 output item 需要 name+namespace 分离形态。
			if _, exists := toolMap[name]; !exists {
				toolMap[name] = responsesToolIdentity{Namespace: ns, Name: realName}
			}
		}
	}
	callID, _ := m["call_id"].(string)
	if callID == "" {
		callID = "call_" + session.NewMessageID()[:16]
	}
	args, _ := m[argField].(string)
	return map[string]any{
		"id":       callID,
		"type":     "function",
		"function": map[string]any{"name": name, "arguments": args},
	}
}

// responsesCustomToolCallOutputItem 映射 {"type":"custom_tool_call_output",call_id,output}
// 为 tool 消息。custom 工具 output 是自由文本（非 JSON 结构），直接透传为 content。
func responsesCustomToolCallOutputItem(m map[string]any) map[string]any {
	callID, _ := m["call_id"].(string)
	if callID == "" {
		return nil
	}
	content, _ := m["output"].(string)
	return map[string]any{"role": "tool", "tool_call_id": callID, "content": content}
}

// responsesFunctionCallOutputItem 映射 {"type":"function_call_output",call_id,output}
// 为 tool 消息。output 可能是字符串或结构化对象（Responses 允许对象形态）——
// 上游 tool 消息 content 只认字符串，对象统一 JSON 序列化。
func responsesFunctionCallOutputItem(m map[string]any) map[string]any {
	callID, _ := m["call_id"].(string)
	if callID == "" {
		return nil // 无 call_id 的 output 无法配对，跳过（payload.go 清理会兜底孤儿）
	}
	var content string
	switch out := m["output"].(type) {
	case string:
		content = out
	case nil:
		content = ""
	default:
		// 对象/数组/数字等结构化输出：JSON 序列化为字符串（保真，客户端可再解析）。
		if raw, err := json.Marshal(out); err == nil {
			content = string(raw)
		}
	}
	return map[string]any{"role": "tool", "tool_call_id": callID, "content": content}
}

// ---------------------------------------------------------------------------
// handler
// ---------------------------------------------------------------------------

// responsesPipeSink Responses 前端的协议差异注入点：错误信封为 Responses 形态
// （{"error":{message,type,code}}，不带 gateway_hint——hint 是 chat 协议的私有
// 扩展字段，Codex 对未知字段不敏感但保持信封纯净）。
type responsesPipeSink struct {
	h         *Handler
	w         http.ResponseWriter
	bareModel string
	// reqFields 非流式响应回显请求字段（P2-2）：parallel_tool_calls/tool_choice
	// 是请求语义的镜像，硬编码假值会误导客户端按错误前提继续会话。
	reqFields responsesReqFields
	// toolMap namespace 工具映射表（拼接名 → 真实身份，ResponsesToChat 处理请求时
	// 生成）：响应侧回译 function_call 的 name/namespace 用。nil = 请求无 namespace
	// 工具，回译全走原样透传（零回归路径）。
	toolMap map[string]responsesToolIdentity
}

func (s responsesPipeSink) writer() http.ResponseWriter { return s.w }

func (responsesPipeSink) writeEndError(w http.ResponseWriter, status int, code, msg, hint string) {
	// 末端透传的 code/msg 已由 pipeline 分类；上游原文若包着业务 JSON 信封，
	// 展开提升（P1-5）——Codex 按 error.code 分类重试，嵌套形态拿不到业务码。
	code, msg = unwindResponsesErrorMessage(code, msg)
	writeResponsesError(w, status, code, msg)
}

func (responsesPipeSink) writeImmediateError(w http.ResponseWriter, status int, code, msg, hint string) {
	// 即时错误（content_blocked / prompt_too_long）的 msg 是上游原文，同样展开提升。
	code, msg = unwindResponsesErrorMessage(code, msg)
	writeResponsesError(w, status, code, msg)
}

// onSyncSuccess 非流式成功：Aggregate 聚合后翻译为 Responses 对象写出。
func (s responsesPipeSink) onSyncSuccess(w http.ResponseWriter, h *Handler, acct *auth.Auth, rc io.ReadCloser, st *chatStat, bareModel string) {
	resp, err := upstream.Aggregate(rc)
	rc.Close()
	if err != nil {
		// 上游流解析失败：与非流式 chat 同语义 502，Responses 错误信封。
		writeResponsesError(w, http.StatusBadGateway, "upstream_parse", err.Error())
		st.status = http.StatusBadGateway
		return
	}
	writeJSON(w, http.StatusOK, responsesSyncToResponseWithToolMap(resp, bareModel, s.reqFields, s.toolMap))
	st.status = http.StatusOK
	if u, ok := resp["usage"].(map[string]any); ok {
		if v, ok := u["completion_tokens"].(float64); ok {
			st.toks = int(v)
		}
		// 成本账本：与 chat 非流式同一口径（usageCreditTotal——credit 存在**且**
		// 至少一边 token 存在才记账；缺失≠0，防止畸形 usage 把 total=0 写进账本）。
		if credit, total, ok := usageCreditTotal(resp); ok {
			h.cfg.Pool.NoteModelCost(acct.UID, bareModel, credit, total)
		}
	}
}

// responses OpenAI Responses API 端点：翻译请求 → 共享轮转管线 → 翻译响应。
func (h *Handler) responses(w http.ResponseWriter, r *http.Request) {
	// 请求体读取/调试落盘与 chat 完全共用（pipeline.go readChatBody）。
	body, status, code, msg := h.readChatBody(r)
	if status != 0 {
		writeResponsesError(w, status, code, msg)
		return
	}

	// 请求翻译：Responses → chat body。翻译失败是客户端畸形请求（非账号问题），
	// 直接 400 终止，不进轮转。toolMap 是 namespace 工具展开时生成的映射表
	// （拼接名 → 真实身份），随 sink 下沉到流式/非流式响应回译。
	chatBody, toolMap, err := responsesToChatWithToolMap(body)
	if err != nil {
		writeResponsesError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	// 翻译后 body 的 peek/stream/model/轮转语义与 chat 前端一致；model 含 realm
	// 前缀（翻译层原样保留，resolveModel 在管线外由本 handler 剥离，顺序与
	// chatCompletions 相同：都先 peek 再 resolve）。
	var peek struct {
		Stream bool   `json:"stream"`
		Model  string `json:"model"`
	}
	_ = json.Unmarshal(chatBody, &peek)
	realm, bareModel := resolveModel(peek.Model)

	// 请求级统计：与 chat 共用同一表格日志；mode 标 "resp-stream"/"resp-sync"
	// 便于运维区分协议来源（stream 字段语义与 chat 相同）。
	st := newChatStat(time.Now(), chatBody, peek.Stream)
	if peek.Stream {
		st.mode = "resp-stream"
	} else {
		st.mode = "resp-sync"
	}
	defer st.done()

	// gateway_hint 判定所需的请求形态（image_url part）：**有意对翻译后的 chat body
	// 判定**而非原始 Responses body——多模态映射发生在 ResponsesToChat 内（image_url
	// part 由翻译层构造），翻译后判定可直接复用 chat 的 hasImagePart 单一来源，且与
	// 管线内 chat body 的实际出站形态一致（不改语义，只选了更准确的判定点）。
	reqHasImage := hasImagePart(chatBody)

	// 非流式回显字段（P2-2）：tool_choice 从翻译后 chat body 取（翻译时透传原值）；
	// parallel_tool_calls 从原始 Responses 请求取（翻译层不透传该字段，须回读原文）。
	fields := responsesReqFields{}
	if tc, ok := chatBodyToolChoice(chatBody); ok {
		fields.ToolChoice = tc
	}
	var rawReq struct {
		ParallelToolCalls *bool `json:"parallel_tool_calls"`
	}
	if json.Unmarshal(body, &rawReq) == nil && rawReq.ParallelToolCalls != nil {
		fields.ParallelToolCalls = rawReq.ParallelToolCalls
	}

	// 共享轮转管线：粘性（metadata.conversation_id 已在翻译时映射）、选号、
	// 熔断、成本账本、错误策略与 chatCompletions 完全同语义。
	sink := responsesPipeSink{h: h, w: w, bareModel: bareModel, reqFields: fields, toolMap: toolMap}
	h.runChatPipeline(r, chatBody, peek.Model, bareModel, realm, peek.Stream, reqHasImage, st, sink)
	// runChatPipeline 已按 sink 回调写出响应（流式/非流式/错误），此处无收尾。
}

// responsesReqFields 非流式响应需要回显的请求字段（P2-2）。ToolChoice 空串表示
// 请求未携带（回显 "auto" 官方缺省）；ParallelToolCalls nil 表示未携带（同样回显
// 缺省 true——Responses 规范默认允许并行）。
type responsesReqFields struct {
	ToolChoice        string
	ParallelToolCalls *bool
}

// chatBodyToolChoice 从翻译后 chat body 提取 tool_choice 字符串形态（对象形态
// 置空串——回显时按缺省 "auto" 处理，与 Responses 客户端的期望一致）。
func chatBodyToolChoice(body []byte) (string, bool) {
	var obj struct {
		ToolChoice any `json:"tool_choice"`
	}
	if json.Unmarshal(body, &obj) != nil {
		return "", false
	}
	if s, ok := obj.ToolChoice.(string); ok {
		return s, true
	}
	return "", false
}

// unwindResponsesErrorMessage 非流式错误路径的双重编码提升（P1-5）：上游 body 原文
// 若本身是 JSON 信封（{"code":6004,"msg":"..."}），提取 code/msg 作为信封字段——
// 与流式 handleFrame 的 unwindNestedErrorMessage 同一语义（那边是 map 原地改，
// 这边是字符串入出，供 sink 回调签名直接使用）。
// 非双重编码形态原样返回（code/msg 零改动）。
func unwindResponsesErrorMessage(code, msg string) (string, string) {
	trimmed := strings.TrimSpace(msg)
	if trimmed == "" || trimmed[0] != '{' {
		return code, msg
	}
	var inner map[string]any
	if json.Unmarshal([]byte(trimmed), &inner) != nil {
		return code, msg
	}
	// 与流式版 unwindNestedErrorMessage（responses_stream.go）同守卫：外层 code 是
	// pipeline 的本地分类名（rate_limit_exceeded 等，Codex 据此重试分类），仅当
	// 外层缺失时才提升内层业务码——无条件覆盖会让同一错误两路径分类分叉。
	if code == "" {
		if c, ok := inner["code"]; ok {
			code = fmt.Sprintf("%v", c)
		}
	}
	if m, ok := inner["msg"].(string); ok && m != "" {
		msg = m
	} else if m, ok := inner["message"].(string); ok && m != "" {
		msg = m
	}
	return code, msg
}

// responsesSyncToResponse responsesSyncToResponseWithToolMap 的兼容入口（既有测试用；
// toolMap 为 nil → 工具名原样透传，与无 namespace 工具的请求语义一致）。
func responsesSyncToResponse(resp map[string]any, model string, reqFields responsesReqFields) map[string]any {
	return responsesSyncToResponseWithToolMap(resp, model, reqFields, nil)
}

// responsesSyncToResponseWithToolMap 把 Aggregate 聚合的 chat completion 翻译为
// Responses 非流式响应对象（stream=false 路径）。
// 输出映射：choices[0].message → output items（reasoning → reasoning item（summary
// part 携带完整思维链文本，与流式收口同形态，P2-3）、content → message item、
// tool_calls → function_call items）；usage 重命名 input_tokens/output_tokens
// （Responses 规范）并做 total_tokens 兜底合成（P0-2，Codex 必填），credit 等
// 上游私有字段原样透出（成本观测对客户端无害且排障有用）。
// toolMap 回译：chat tool_call 的 function.name 若命中 namespace 映射表，输出的
// function_call item name 还原为真实名、namespace 填真实命名空间——Codex 按
// namespace 字段路由到 multi_agent 处理器，拼接名直透会让它找不到对应工具。
// 未命中映射 → name 原样、namespace 字段省略。
// parallel_tool_calls/tool_choice 回显请求实际值（P2-2）。
func responsesSyncToResponseWithToolMap(resp map[string]any, model string, reqFields responsesReqFields, toolMap map[string]responsesToolIdentity) map[string]any {
	respID := "resp_" + session.NewMessageID()
	output := make([]any, 0, 3)
	finish := "stop"
	if chs, ok := resp["choices"].([]any); ok && len(chs) > 0 {
		if c, ok := chs[0].(map[string]any); ok {
			if fr, ok := c["finish_reason"].(string); ok && fr != "" {
				finish = fr
			}
			msg, _ := c["message"].(map[string]any)
			if msg != nil {
				if rc, _ := msg["reasoning_content"].(string); rc != "" {
					// P2-3：summary part 带完整思维链文本（与流式收口同形态）——
					// 恒空 summary 会让 Codex UI 的思维链面板永远空白。
					output = append(output, map[string]any{
						"type": "reasoning", "id": "rs_" + session.NewMessageID(),
						"summary": []any{map[string]any{"type": "summary_text", "text": rc}},
					})
				}
				if txt, _ := msg["content"].(string); txt != "" {
					output = append(output, map[string]any{
						"type":    "message",
						"id":      "msg_" + session.NewMessageID(),
						"role":    "assistant",
						"status":  "completed",
						"content": []any{map[string]any{"type": "output_text", "text": txt, "annotations": []any{}}},
					})
				}
				if tcs, ok := msg["tool_calls"].([]any); ok {
					for _, tc := range tcs {
						tcm, ok := tc.(map[string]any)
						if !ok {
							continue
						}
						fn, _ := tcm["function"].(map[string]any)
						name, _ := fn["name"].(string)
						args, _ := fn["arguments"].(string)
						callID, _ := tcm["id"].(string)
						fc := map[string]any{
							"type":      "function_call",
							"id":        "fc_" + session.NewMessageID(),
							"call_id":   callID,
							"name":      name,
							"arguments": args,
							"status":    "completed",
						}
						// namespace 回译：命中映射表才拆回 (namespace, real_name)；
						// 未命中（普通工具/toolMap 为 nil）保持 name 原样、namespace
						// 字段省略（Responses 的 FunctionCall item 命名空间可选，
						// 缺省即无命名空间语义）。
						if id, hit := toolMap[name]; hit {
							fc["name"] = id.Name
							fc["namespace"] = id.Namespace
						}
						output = append(output, fc)
					}
				}
			}
		}
	}
	out := map[string]any{
		"id":         respID,
		"object":     "response",
		"created_at": time.Now().Unix(),
		"status":     "completed",
		"model":      model,
		"output":     output,
		"tools":      []any{},
	}
	// 回显请求实际值（P2-2）：未携带时回落官方缺省（parallel 允许 / auto）。
	if reqFields.ParallelToolCalls != nil {
		out["parallel_tool_calls"] = *reqFields.ParallelToolCalls
	} else {
		out["parallel_tool_calls"] = true
	}
	if reqFields.ToolChoice != "" {
		out["tool_choice"] = reqFields.ToolChoice
	} else {
		out["tool_choice"] = "auto"
	}
	if finish == "length" {
		out["status"] = "incomplete"
		out["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
	}
	if usage, ok := resp["usage"].(map[string]any); ok {
		// P0-2：total_tokens 是 Codex 的必填字段，与流式 completed 同一兜底实现
		// （upstream.EnsureUsageTotal，input+output 合成，缺一不编造）。
		out["usage"] = responsesUsage(upstream.EnsureUsageTotal(usage))
	} else {
		out["usage"] = nil
	}
	return out
}

// responsesUsage 把 chat usage 映射为 Responses usage 命名（input_tokens/
// output_tokens/total_tokens），其余字段（credit 等）原样保留。
func responsesUsage(u map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range u {
		switch k {
		case "prompt_tokens":
			out["input_tokens"] = v
		case "completion_tokens":
			out["output_tokens"] = v
		case "total_tokens":
			out["total_tokens"] = v
		default:
			out[k] = v
		}
	}
	return out
}

// writeResponsesError Responses 形态错误信封：{"error":{message,type,code}}。
// code 透传 pipeline 分类（P1-4）：Codex 按 error.code 字符串（rate_limit_exceeded
// 等）做重试分类，恒 null 会让所有错误都被当成不可重试——本地/上游 code 全接进去。
// 不带 gateway_hint（chat 私有扩展；Responses 信封保持规范形态）。
func writeResponsesError(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    responsesErrorType(status, code),
			"code":    code,
		},
	})
}

// responsesErrorType 错误 type 分类：仅**本地**判定的 4xx（invalid_request /
// request_body_too_large）是 invalid_request_error——客户端问题、改请求才有意义；
// 上游 4xx（429 限流、400 内容拦截等）保持 api_error——不是客户端请求形态的错，
// 标 invalid_request_error 会误导 Codex 不重试（上游 429 冷却后可自愈）。
// 判据用 code 白名单而非状态码区间：pipeline 对上游/本地错误共用 4xx 段。
func responsesErrorType(status int, code string) string {
	switch code {
	case "invalid_request", "request_body_too_large", "invalid_api_key":
		return "invalid_request_error"
	}
	return "api_error"
}
