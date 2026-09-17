// responses_test.go /v1/responses 翻译层测试：请求翻译（ResponsesToChat）、
// 流式翻译器（responsesStreamTranslator）、非流式聚合翻译与 handler 集成。
package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"workbuddy2api/internal/auth"
	"workbuddy2api/internal/session"
)

// decodeChatBody 解析翻译后的 chat body（测试辅助）。
func decodeChatBody(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		t.Fatalf("translated body not json: %v body=%s", err, body)
	}
	return obj
}

// TestResponsesToChatStringInput string input → 单条 user 消息。
func TestResponsesToChatStringInput(t *testing.T) {
	out, err := ResponsesToChat([]byte(`{"model":"cn:glm-5.2","input":"hi","stream":true}`))
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	obj := decodeChatBody(t, out)
	msgs, _ := obj["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages len=%d want 1 body=%s", len(msgs), out)
	}
	m := msgs[0].(map[string]any)
	if m["role"] != "user" || m["content"] != "hi" {
		t.Errorf("message=%v want {user hi}", m)
	}
	if obj["model"] != "cn:glm-5.2" {
		t.Errorf("model=%v want cn:glm-5.2（realm 前缀保留）", obj["model"])
	}
	if obj["stream"] != true {
		t.Errorf("stream=%v want true", obj["stream"])
	}
}

// TestResponsesToChatItemsArray input 数组全形态：message user/assistant、
// function_call、function_call_output、reasoning 跳过。
func TestResponsesToChatItemsArray(t *testing.T) {
	body := `{"model":"glm-5.2","input":[
		{"type":"message","role":"user","content":"list files"},
		{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking..."}]},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I will"}]},
		{"type":"function_call","name":"shell","arguments":"{\"cmd\":\"ls\"}","call_id":"call_abc"},
		{"type":"function_call_output","call_id":"call_abc","output":"file1\nfile2"}
	]}`
	out, err := ResponsesToChat([]byte(body))
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	obj := decodeChatBody(t, out)
	msgs, _ := obj["messages"].([]any)
	if len(msgs) != 4 {
		t.Fatalf("messages len=%d want 4（reasoning 跳过） body=%s", len(msgs), out)
	}
	// user 直传
	m0 := msgs[0].(map[string]any)
	if m0["role"] != "user" || m0["content"] != "list files" {
		t.Errorf("m0=%v", m0)
	}
	// assistant parts → 拼接 text
	m1 := msgs[1].(map[string]any)
	if m1["role"] != "assistant" || m1["content"] != "I will" {
		t.Errorf("m1=%v want assistant/I will", m1)
	}
	// function_call → assistant tool_calls
	m2 := msgs[2].(map[string]any)
	tcs, _ := m2["tool_calls"].([]any)
	if len(tcs) != 1 {
		t.Fatalf("m2 tool_calls=%v", m2)
	}
	tc := tcs[0].(map[string]any)
	if tc["id"] != "call_abc" || tc["type"] != "function" {
		t.Errorf("tc=%v", tc)
	}
	fn := tc["function"].(map[string]any)
	if fn["name"] != "shell" || fn["arguments"] != `{"cmd":"ls"}` {
		t.Errorf("fn=%v", fn)
	}
	// function_call_output → tool 消息
	m3 := msgs[3].(map[string]any)
	if m3["role"] != "tool" || m3["tool_call_id"] != "call_abc" || m3["content"] != "file1\nfile2" {
		t.Errorf("m3=%v", m3)
	}
}

// TestResponsesToChatInstructions reasoning.effort / instructions / max_output_tokens。
func TestResponsesToChatInstructions(t *testing.T) {
	body := `{"model":"glm-5.2","instructions":"be terse","input":"hi",
		"reasoning":{"effort":"high"},"max_output_tokens":1024,"temperature":0.5,"top_p":0.9}`
	out, err := ResponsesToChat([]byte(body))
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	obj := decodeChatBody(t, out)
	msgs, _ := obj["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages len=%d want 2（instructions 最前） body=%s", len(msgs), out)
	}
	m0 := msgs[0].(map[string]any)
	if m0["role"] != "system" || m0["content"] != "be terse" {
		t.Errorf("m0=%v want system/be terse", m0)
	}
	if obj["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort=%v want high", obj["reasoning_effort"])
	}
	if obj["max_tokens"] != float64(1024) {
		t.Errorf("max_tokens=%v want 1024", obj["max_tokens"])
	}
	if obj["temperature"] != 0.5 || obj["top_p"] != 0.9 {
		t.Errorf("sampling=%v", obj)
	}
	// Responses 专有字段不得透传。
	for _, k := range []string{"instructions", "reasoning", "max_output_tokens", "previous_response_id", "store", "truncation", "text"} {
		if _, has := obj[k]; has {
			t.Errorf("field %q must not leak to chat body", k)
		}
	}
}

// TestResponsesToChatToolsNested tools 扁平 → 嵌套；strict 删除；嵌套形态原样。
func TestResponsesToChatToolsNested(t *testing.T) {
	body := `{"model":"glm-5.2","input":"hi","tools":[
		{"type":"function","name":"shell","description":"run","parameters":{"type":"object"},"strict":true},
		{"type":"function","function":{"name":"nested","description":"already nested"}}
	]}`
	out, err := ResponsesToChat([]byte(body))
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	obj := decodeChatBody(t, out)
	tools, _ := obj["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools len=%d body=%s", len(tools), out)
	}
	t0 := tools[0].(map[string]any)
	fn, ok := t0["function"].(map[string]any)
	if !ok {
		t.Fatalf("flat tool not nested: %v", t0)
	}
	if fn["name"] != "shell" || fn["description"] != "run" {
		t.Errorf("fn=%v", fn)
	}
	if _, has := fn["strict"]; has {
		t.Errorf("strict must be dropped: %v", fn)
	}
	if _, has := t0["name"]; has {
		t.Errorf("flat name key must not remain: %v", t0)
	}
	// 已嵌套形态原样保留。
	t1 := tools[1].(map[string]any)
	fn1, _ := t1["function"].(map[string]any)
	if fn1["name"] != "nested" {
		t.Errorf("nested tool altered: %v", t1)
	}
}

// TestResponsesToChatConversationMapping metadata.conversation / 顶层 conversation
// → metadata.conversation_id（session.ExtractKey 粘性键）。
func TestResponsesToChatConversationMapping(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"meta-object", `{"model":"m","input":"hi","metadata":{"conversation":{"id":"conv-1"}}}`},
		{"meta-string", `{"model":"m","input":"hi","metadata":{"conversation":"conv-2"}}`},
		{"top-level", `{"model":"m","input":"hi","conversation":"conv-3"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := ResponsesToChat([]byte(tc.body))
			if err != nil {
				t.Fatalf("ResponsesToChat: %v", err)
			}
			obj := decodeChatBody(t, out)
			meta, ok := obj["metadata"].(map[string]any)
			if !ok {
				t.Fatalf("metadata missing: %s", out)
			}
			if meta["conversation_id"] == "" {
				t.Errorf("conversation_id empty: %s", out)
			}
			// 粘性键必须能被 session.ExtractKey 命中（端到端口径）。
			if key := session.ExtractKey(out); key == "" {
				t.Errorf("session.ExtractKey misses translated body: %s", out)
			}
			// 原始 conversation 键不得残留。
			if _, has := obj["conversation"]; has {
				t.Errorf("top-level conversation leaked: %s", out)
			}
		})
	}
}

// TestResponsesToChatImagePart input_image → chat image_url part（多模态通路）。
func TestResponsesToChatImagePart(t *testing.T) {
	body := `{"model":"glm-5.2","input":[{"type":"message","role":"user","content":[
		{"type":"input_text","text":"what is this"},
		{"type":"input_image","image_url":"https://x.test/a.png"}
	]}]}`
	out, err := ResponsesToChat([]byte(body))
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	if !hasImagePart(out) {
		t.Fatalf("translated body lost image part: %s", out)
	}
	obj := decodeChatBody(t, out)
	msgs, _ := obj["messages"].([]any)
	m := msgs[0].(map[string]any)
	parts, _ := m["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts=%v", parts)
	}
	img := parts[1].(map[string]any)
	if img["type"] != "image_url" || img["image_url"] != "https://x.test/a.png" {
		t.Errorf("img part=%v", img)
	}
}

// TestResponsesToChatFunctionCallOutputObject output 为对象 → JSON 序列化字符串。
func TestResponsesToChatFunctionCallOutputObject(t *testing.T) {
	body := `{"model":"m","input":[{"type":"function_call_output","call_id":"c1","output":{"rows":3,"ok":true}}]}`
	out, err := ResponsesToChat([]byte(body))
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	obj := decodeChatBody(t, out)
	msgs, _ := obj["messages"].([]any)
	m := msgs[0].(map[string]any)
	if m["content"] != `{"ok":true,"rows":3}` && m["content"] != `{"rows":3,"ok":true}` {
		t.Errorf("object output not serialized: %v", m)
	}
}

// TestResponsesToChatConsecutiveToolCallsMerged 并行工具调用合并（11148 修复）：
// 连续 function_call item 必须落在同一条 assistant 消息的 tool_calls 里；reasoning
// item 夹在中间不冲刷（Codex 历史里 reasoning 与其发起的 call 同轮相邻）；message/
// function_call_output 出现时冲刷缓冲。
func TestResponsesToChatConsecutiveToolCallsMerged(t *testing.T) {
	body := `{"model":"m","input":[` +
		`{"type":"message","role":"user","content":"do it"},` +
		`{"type":"function_call","call_id":"c1","name":"shell","arguments":"{\"a\":1}"},` +
		`{"type":"reasoning","summary":[]},` +
		`{"type":"function_call","call_id":"c2","name":"apply_patch","arguments":"patch"},` +
		`{"type":"custom_tool_call","call_id":"c3","name":"custom","input":"raw-in"},` +
		`{"type":"function_call_output","call_id":"c1","output":"ok"}]}`
	out, err := ResponsesToChat([]byte(body))
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	msgs, _ := decodeChatBody(t, out)["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages=%d want 3 (user / merged-assistant / tool)", len(msgs))
	}
	asst := msgs[1].(map[string]any)
	calls, _ := asst["tool_calls"].([]any)
	if len(calls) != 3 {
		t.Fatalf("merged tool_calls=%d want 3", len(calls))
	}
	// 条目形态与 id 顺序校验（c1/c2 function + c3 custom→function）。
	wantIDs := []string{"c1", "c2", "c3"}
	wantArgs := []string{`{"a":1}`, "patch", "raw-in"}
	for i, c := range calls {
		cm := c.(map[string]any)
		if cm["id"] != wantIDs[i] {
			t.Errorf("call[%d].id=%v want %s", i, cm["id"], wantIDs[i])
		}
		fn, _ := cm["function"].(map[string]any)
		if fn["arguments"] != wantArgs[i] {
			t.Errorf("call[%d].arguments=%v want %q", i, fn["arguments"], wantArgs[i])
		}
	}
	if msgs[2].(map[string]any)["tool_call_id"] != "c1" {
		t.Errorf("third message not the tool output")
	}
}

// --- 流式翻译器 ---

// respEvents 解析翻译器产出的 SSE 事件流为 (event, data) 序列。
func respEvents(t *testing.T, raw string) [][2]string {
	t.Helper()
	var out [][2]string
	for _, block := range strings.Split(raw, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var ev, data string
		for _, line := range strings.Split(block, "\n") {
			if strings.HasPrefix(line, "event: ") {
				ev = strings.TrimPrefix(line, "event: ")
			} else if strings.HasPrefix(line, "data: ") {
				data = strings.TrimPrefix(line, "data: ")
			}
		}
		if ev != "" {
			out = append(out, [2]string{ev, data})
		}
	}
	return out
}

// runTranslator 喂上游 SSE 帧给翻译器，返回客户端侧收到的原始输出。
func runTranslator(t *testing.T, upstreamFrames string, model string) string {
	t.Helper()
	rec := httptest.NewRecorder()
	tr := newResponsesStreamTranslator(rec, model)
	if err := tr.run(strings.NewReader(upstreamFrames)); err != nil {
		t.Fatalf("translator run: %v", err)
	}
	return rec.Body.String()
}

// fullStream 覆盖 reasoning + content + tool_calls 分片 + usage + [DONE] 的上游流。
const fullStream = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"think\"}}]}\n\n" +
	"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"ing\"}}]}\n\n" +
	"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n" +
	"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n" +
	"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"{\\\"cm\"}}]}}]}\n\n" +
	"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"g\\\":1}\"}}]}}]}\n\n" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15,\"credit\":0.05}}\n\n" +
	"data: [DONE]\n\n"

// TestResponsesStreamFullSequence 全形态事件流断言：首事件 created、末事件
// completed、配对完整、delta 拼接还原、usage 映射。
func TestResponsesStreamFullSequence(t *testing.T) {
	raw := runTranslator(t, fullStream, "glm-5.2")
	evs := respEvents(t, raw)
	if len(evs) == 0 {
		t.Fatalf("no events: %s", raw)
	}
	if evs[0][0] != "response.created" {
		t.Errorf("first event=%s want response.created", evs[0][0])
	}
	if evs[len(evs)-1][0] != "response.completed" {
		t.Errorf("last event=%s want response.completed", evs[len(evs)-1][0])
	}
	// sequence_number 严格递增。
	lastSeq := int64(0)
	for i, e := range evs {
		var obj map[string]any
		if err := json.Unmarshal([]byte(e[1]), &obj); err != nil {
			t.Fatalf("event %d data not json: %v", i, err)
		}
		seq := obj["sequence_number"].(float64)
		if int64(seq) != lastSeq+1 {
			t.Errorf("event %d seq=%d want %d", i, int64(seq), lastSeq+1)
		}
		lastSeq = int64(seq)
		if obj["type"] != e[0] {
			t.Errorf("event %d type mismatch data.type=%v event=%s", i, obj["type"], e[0])
		}
	}
	// 事件类型集合与配对。
	var types []string
	for _, e := range evs {
		types = append(types, e[0])
	}
	expectContains := []string{
		"response.output_item.added",            // reasoning 收口
		"response.reasoning_summary_part.added", // P2-4：summary part 形态
		"response.reasoning_summary_text.done",  // 一次性完整文本（Codex 可见形态）
		"response.output_item.added",            // message
		"response.content_part.added",           // part
		"response.output_text.delta",            // 2 次
		"response.output_item.added",            // function_call
		"response.function_call_arguments.delta",
		"response.function_call_arguments.done",
		"response.output_item.done",
		"response.completed",
	}
	for _, want := range expectContains {
		found := false
		for _, ty := range types {
			if ty == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("missing event %q in %v", want, types)
		}
	}
	// added/done 计数配对：3 个 item（reasoning/message/function_call）。
	added, done := 0, 0
	for _, ty := range types {
		switch ty {
		case "response.output_item.added":
			added++
		case "response.output_item.done":
			done++
		}
	}
	if added != 3 || done != 3 {
		t.Errorf("item added=%d done=%d want 3/3", added, done)
	}
	// 正文 delta 拼接还原全文。
	var text strings.Builder
	for _, e := range evs {
		if e[0] != "response.output_text.delta" {
			continue
		}
		var obj map[string]any
		_ = json.Unmarshal([]byte(e[1]), &obj)
		text.WriteString(obj["delta"].(string))
	}
	if text.String() != "Hello world" {
		t.Errorf("text deltas=%q want %q", text.String(), "Hello world")
	}
	// arguments delta 拼接还原。
	var args strings.Builder
	for _, e := range evs {
		if e[0] != "response.function_call_arguments.delta" {
			continue
		}
		var obj map[string]any
		_ = json.Unmarshal([]byte(e[1]), &obj)
		args.WriteString(obj["delta"].(string))
	}
	if args.String() != `{"cmg":1}` {
		t.Errorf("args deltas=%q want {\"cmg\":1}", args.String())
	}
	// completed 事件：usage 映射 + output items 全量。
	var comp struct {
		Response struct {
			Status string `json:"status"`
			Usage  struct {
				InputTokens  any `json:"input_tokens"`
				OutputTokens any `json:"output_tokens"`
				TotalTokens  any `json:"total_tokens"`
				Credit       any `json:"credit"`
			} `json:"usage"`
			Output []map[string]any `json:"output"`
		} `json:"response"`
	}
	last := evs[len(evs)-1]
	if err := json.Unmarshal([]byte(last[1]), &comp); err != nil {
		t.Fatalf("completed data: %v", err)
	}
	if comp.Response.Status != "completed" {
		t.Errorf("status=%s want completed", comp.Response.Status)
	}
	if comp.Response.Usage.InputTokens != float64(10) || comp.Response.Usage.OutputTokens != float64(5) {
		t.Errorf("usage mapping wrong: %+v", comp.Response.Usage)
	}
	if comp.Response.Usage.TotalTokens != float64(15) {
		t.Errorf("total_tokens=%v want 15", comp.Response.Usage.TotalTokens)
	}
	if comp.Response.Usage.Credit == nil {
		t.Errorf("credit must pass through: %+v", comp.Response.Usage)
	}
	// output 顺序：reasoning(0) → message(1) → function_call(2)。
	if len(comp.Response.Output) != 3 {
		t.Fatalf("completed output len=%d want 3", len(comp.Response.Output))
	}
	if comp.Response.Output[0]["type"] != "reasoning" {
		t.Errorf("output[0]=%v want reasoning", comp.Response.Output[0])
	}
	if comp.Response.Output[1]["type"] != "message" {
		t.Errorf("output[1]=%v want message", comp.Response.Output[1])
	}
	fc := comp.Response.Output[2]
	if fc["type"] != "function_call" || fc["name"] != "shell" || fc["call_id"] != "call_a" {
		t.Errorf("output[2]=%v want function_call/shell/call_a", fc)
	}
	if fc["arguments"] != `{"cmg":1}` {
		t.Errorf("fc arguments=%v want full json", fc["arguments"])
	}
}

// TestResponsesStreamFinishLength finish_reason=length → status=incomplete +
// incomplete_details.max_output_tokens。
func TestResponsesStreamFinishLength(t *testing.T) {
	frames := "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n" +
		"data: [DONE]\n\n"
	raw := runTranslator(t, frames, "m")
	evs := respEvents(t, raw)
	var comp struct {
		Response struct {
			Status            string           `json:"status"`
			IncompleteDetails map[string]any   `json:"incomplete_details"`
			Output            []map[string]any `json:"output"`
		} `json:"response"`
	}
	if err := json.Unmarshal([]byte(evs[len(evs)-1][1]), &comp); err != nil {
		t.Fatalf("completed data: %v", err)
	}
	if comp.Response.Status != "incomplete" {
		t.Errorf("status=%s want incomplete", comp.Response.Status)
	}
	if comp.Response.IncompleteDetails["reason"] != "max_output_tokens" {
		t.Errorf("incomplete_details=%v", comp.Response.IncompleteDetails)
	}
	if len(comp.Response.Output) != 1 || comp.Response.Output[0]["type"] != "message" {
		t.Errorf("output=%v want single message", comp.Response.Output)
	}
}

// TestResponsesStreamUsageMissing usage 缺失 → completed.usage = null（不编造）。
func TestResponsesStreamUsageMissing(t *testing.T) {
	frames := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	raw := runTranslator(t, frames, "m")
	if !strings.Contains(raw, `"usage":null`) {
		t.Errorf("usage must be null when absent: %s", raw)
	}
}

// TestResponsesStreamErrorFrame 上游 error 帧 → response.failed 终止（无 completed）。
func TestResponsesStreamErrorFrame(t *testing.T) {
	frames := "data: {\"error\":{\"message\":\"boom\",\"code\":6004}}\n\n"
	raw := runTranslator(t, frames, "m")
	evs := respEvents(t, raw)
	if evs[len(evs)-1][0] != "response.failed" {
		t.Fatalf("last event=%s want response.failed body=%s", evs[len(evs)-1][0], raw)
	}
	var obj struct {
		Response struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		} `json:"response"`
	}
	_ = json.Unmarshal([]byte(evs[len(evs)-1][1]), &obj)
	if obj.Response.Error.Message != "boom" {
		t.Errorf("error message=%v want boom", obj.Response.Error.Message)
	}
	for _, e := range evs {
		if e[0] == "response.completed" {
			t.Errorf("completed must not follow failed: %s", raw)
		}
	}
}

// TestResponsesStreamEmpty 上游空流 → response.failed（upstream_parse 语义）。
func TestResponsesStreamEmpty(t *testing.T) {
	rec := httptest.NewRecorder()
	tr := newResponsesStreamTranslator(rec, "m")
	err := tr.run(strings.NewReader("data: [DONE]\n\n"))
	if err == nil {
		t.Fatalf("empty stream must error")
	}
	raw := rec.Body.String()
	evs := respEvents(t, raw)
	// P1-3：created 恒为首事件（不再等首个有效帧），空流场景是 created → failed。
	if len(evs) != 2 || evs[0][0] != "response.created" || evs[len(evs)-1][0] != "response.failed" {
		t.Fatalf("events=%v want [created failed]", evs)
	}
	if !strings.Contains(raw, "upstream_parse") {
		t.Errorf("failed event must carry upstream_parse: %s", raw)
	}
}

// TestResponsesStreamToolCallNoIndex 缺 index 的 tool_call 按 id 归位（跨帧延续，
// 与 sse.go mergeToolCallsChunk 同策略）。
func TestResponsesStreamToolCallNoIndex(t *testing.T) {
	frames := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_x\",\"type\":\"function\",\"function\":{\"name\":\"f1\",\"arguments\":\"a\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_x\",\"function\":{\"arguments\":\"b\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"
	raw := runTranslator(t, frames, "m")
	evs := respEvents(t, raw)
	added := 0
	for _, e := range evs {
		if e[0] == "response.output_item.added" {
			added++
		}
	}
	if added != 1 {
		t.Errorf("same-id fragments must merge into one item: added=%d body=%s", added, raw)
	}
	if !strings.Contains(raw, `"arguments":"ab"`) {
		t.Errorf("arguments not merged: %s", raw)
	}
}

// --- namespace 工具（Codex multi_agent 委派）---

// TestResponsesToChatNamespaceTools namespace 工具展开：multi_agent_v1 包（1 function
// + 1 custom 子工具）+ 普通 function + web_search → chat tools 只剩 3 条 function
// 形态（拼接名、custom parameters 兜底、web_search 剔除），toolMap 登记 2 条映射。
func TestResponsesToChatNamespaceTools(t *testing.T) {
	body := `{"model":"m","input":"hi","tools":[
		{"type":"namespace","name":"multi_agent_v1","description":"Tools for spawning and managing sub-agents.","tools":[
			{"type":"function","name":"spawn_agent","description":"spawn a sub agent","parameters":{"type":"object","properties":{"task":{"type":"string"}},"required":["task"]}},
			{"type":"custom","name":"apply_patch","description":"freeform patch"}
		]},
		{"type":"function","name":"shell","description":"run","parameters":{"type":"object"}},
		{"type":"web_search"}
	]}`
	out, toolMap, err := responsesToChatWithToolMap([]byte(body))
	if err != nil {
		t.Fatalf("responsesToChatWithToolMap: %v", err)
	}
	obj := decodeChatBody(t, out)
	tools, _ := obj["tools"].([]any)
	if len(tools) != 3 {
		t.Fatalf("tools len=%d want 3 (2 展开自 namespace + 1 function, web_search 剔除) body=%s", len(tools), out)
	}
	// 子工具 1：function 展开为拼接名，description 附加 namespace 前缀，parameters 透传。
	fn0 := tools[0].(map[string]any)["function"].(map[string]any)
	if fn0["name"] != "multi_agent_v1spawn_agent" {
		t.Errorf("fn0.name=%v want multi_agent_v1spawn_agent（Codex tool_name.rs Display：直接拼接）", fn0["name"])
	}
	if fn0["description"] != "[namespace multi_agent_v1: Tools for spawning and managing sub-agents.] spawn a sub agent" {
		t.Errorf("fn0.description=%v", fn0["description"])
	}
	if _, ok := fn0["parameters"].(map[string]any); !ok {
		t.Errorf("fn0.parameters lost: %v", fn0)
	}
	// 子工具 2：custom 子工具 parameters 缺失 → 单字符串 input 兜底。
	fn1 := tools[1].(map[string]any)["function"].(map[string]any)
	if fn1["name"] != "multi_agent_v1apply_patch" {
		t.Errorf("fn1.name=%v want multi_agent_v1apply_patch", fn1["name"])
	}
	params, _ := fn1["parameters"].(map[string]any)
	props, _ := params["properties"].(map[string]any)
	if _, ok := props["input"]; !ok {
		t.Errorf("custom sub-tool parameters must fall back to input: %v", params)
	}
	// 子工具 3：普通 function 原名透传。
	fn2 := tools[2].(map[string]any)["function"].(map[string]any)
	if fn2["name"] != "shell" {
		t.Errorf("fn2.name=%v want shell", fn2["name"])
	}
	// toolMap：2 条映射（拼接名 → namespace + 真实名）。
	if len(toolMap) != 2 {
		t.Fatalf("toolMap len=%d want 2: %v", len(toolMap), toolMap)
	}
	if id := toolMap["multi_agent_v1spawn_agent"]; id.Namespace != "multi_agent_v1" || id.Name != "spawn_agent" {
		t.Errorf("toolMap[spawn]=%+v", id)
	}
	if id := toolMap["multi_agent_v1apply_patch"]; id.Namespace != "multi_agent_v1" || id.Name != "apply_patch" {
		t.Errorf("toolMap[apply_patch]=%+v", id)
	}
}

// TestResponsesStreamNamespaceToolCallIdentity 流式回译：上游 tool_call 的拼接名
// multi_agent_v1spawn_agent → output_item.done 的 item.name=spawn_agent、
// item.namespace=multi_agent_v1（added 与 completed.output 同口径）。
func TestResponsesStreamNamespaceToolCallIdentity(t *testing.T) {
	frames := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_ns\",\"type\":\"function\",\"function\":{\"name\":\"multi_agent_v1spawn_agent\",\"arguments\":\"{\\\"task\\\":\\\"hi\\\"}\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"
	rec := httptest.NewRecorder()
	tr := newResponsesStreamTranslator(rec, "m")
	tr.toolMap = map[string]responsesToolIdentity{
		"multi_agent_v1spawn_agent": {Namespace: "multi_agent_v1", Name: "spawn_agent"},
	}
	if err := tr.run(strings.NewReader(frames)); err != nil {
		t.Fatalf("translator run: %v", err)
	}
	raw := rec.Body.String()
	evs := respEvents(t, raw)
	checkItem := func(item map[string]any, where string) {
		if item["type"] != "function_call" {
			t.Errorf("%s: item type=%v", where, item["type"])
			return
		}
		if item["name"] != "spawn_agent" {
			t.Errorf("%s: name=%v want spawn_agent（拼接名必须拆回）", where, item["name"])
		}
		if item["namespace"] != "multi_agent_v1" {
			t.Errorf("%s: namespace=%v want multi_agent_v1", where, item["namespace"])
		}
		if item["call_id"] != "call_ns" {
			t.Errorf("%s: call_id=%v want call_ns", where, item["call_id"])
		}
	}
	found := 0
	for _, e := range evs {
		if e[0] != "response.output_item.done" && e[0] != "response.output_item.added" {
			continue
		}
		var obj struct {
			Item map[string]any `json:"item"`
		}
		if err := json.Unmarshal([]byte(e[1]), &obj); err != nil {
			t.Fatalf("%s data: %v", e[0], err)
		}
		if obj.Item["type"] == "function_call" {
			checkItem(obj.Item, e[0])
			found++
		}
	}
	if found < 2 {
		t.Fatalf("function_call added/done items=%d want >=2 body=%s", found, raw)
	}
	// completed.output 里的 function_call 同口径回译。
	var comp struct {
		Response struct {
			Output []map[string]any `json:"output"`
		} `json:"response"`
	}
	if err := json.Unmarshal([]byte(evs[len(evs)-1][1]), &comp); err != nil {
		t.Fatalf("completed data: %v", err)
	}
	checkItem(comp.Response.Output[0], "completed.output")
	rec2 := httptest.NewRecorder()
	tr2 := newResponsesStreamTranslator(rec2, "m")
	tr2.toolMap = tr.toolMap
	// 同流两条调用：拼接名命中（回译）+ 普通名未命中（原样、无 namespace）。
	frames2 := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[" +
		"{\"index\":0,\"id\":\"call_plain\",\"type\":\"function\",\"function\":{\"name\":\"multi_agent_v1spawn_agent\"}}," +
		"{\"index\":1,\"id\":\"call_o\",\"type\":\"function\",\"function\":{\"name\":\"shell\"}}" +
		"]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n"
	if err := tr2.run(strings.NewReader(frames2)); err != nil {
		t.Fatalf("translator run 2: %v", err)
	}
	raw2 := rec2.Body.String()
	var comp2 struct {
		Response struct {
			Output []map[string]any `json:"output"`
		} `json:"response"`
	}
	evs2 := respEvents(t, raw2)
	if err := json.Unmarshal([]byte(evs2[len(evs2)-1][1]), &comp2); err != nil {
		t.Fatalf("completed data: %v", err)
	}
	if len(comp2.Response.Output) != 2 {
		t.Fatalf("output=%v", comp2.Response.Output)
	}
	if comp2.Response.Output[0]["name"] != "spawn_agent" || comp2.Response.Output[0]["namespace"] != "multi_agent_v1" {
		t.Errorf("output[0]=%v want spawn_agent/multi_agent_v1", comp2.Response.Output[0])
	}
	if comp2.Response.Output[1]["name"] != "shell" {
		t.Errorf("output[1]=%v want shell 原样", comp2.Response.Output[1])
	}
	if _, has := comp2.Response.Output[1]["namespace"]; has {
		t.Errorf("plain tool must omit namespace: %v", comp2.Response.Output[1])
	}
}

// TestResponsesSyncNamespaceToolCallIdentity 非流式回译：命中映射 → name/namespace
// 拆回；未命中 → name 原样且无 namespace 字段。
func TestResponsesSyncNamespaceToolCallIdentity(t *testing.T) {
	toolMap := map[string]responsesToolIdentity{
		"multi_agent_v1spawn_agent": {Namespace: "multi_agent_v1", Name: "spawn_agent"},
	}
	resp := map[string]any{
		"choices": []any{map[string]any{
			"message": map[string]any{"role": "assistant", "tool_calls": []any{
				map[string]any{"id": "c1", "type": "function",
					"function": map[string]any{"name": "multi_agent_v1spawn_agent", "arguments": "{}"}},
				map[string]any{"id": "c2", "type": "function",
					"function": map[string]any{"name": "shell", "arguments": "{}"}},
			}},
			"finish_reason": "tool_calls",
		}},
	}
	out := responsesSyncToResponseWithToolMap(resp, "m", responsesReqFields{}, toolMap)
	output, _ := out["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output=%v", output)
	}
	fc0 := output[0].(map[string]any)
	if fc0["name"] != "spawn_agent" || fc0["namespace"] != "multi_agent_v1" || fc0["call_id"] != "c1" {
		t.Errorf("fc0=%v want spawn_agent/multi_agent_v1/c1", fc0)
	}
	fc1 := output[1].(map[string]any)
	if fc1["name"] != "shell" {
		t.Errorf("fc1=%v want shell 原样", fc1)
	}
	if _, has := fc1["namespace"]; has {
		t.Errorf("未命中映射不得带 namespace 字段: %v", fc1)
	}
}

// TestResponsesToChatHistoryFunctionCallNamespace history 回传拼接：Codex 下一轮把
// 上轮 function_call item（name=spawn_agent + namespace=multi_agent_v1）原样放回
// input，chat 侧工具名必须拼回 multi_agent_v1spawn_agent（与上轮模型看到的拼接名
// 恒等），配对 output 不受影响；namespace=="functions" 不拼接。
func TestResponsesToChatHistoryFunctionCallNamespace(t *testing.T) {
	body := `{"model":"m","input":[
		{"type":"message","role":"user","content":"delegate"},
		{"type":"function_call","name":"spawn_agent","namespace":"multi_agent_v1","arguments":"{\"task\":\"x\"}","call_id":"call_n1"},
		{"type":"function_call_output","call_id":"call_n1","output":"spawned"},
		{"type":"function_call","name":"plain","namespace":"functions","arguments":"{}","call_id":"call_n2"}
	]}`
	out, toolMap, err := responsesToChatWithToolMap([]byte(body))
	if err != nil {
		t.Fatalf("responsesToChatWithToolMap: %v", err)
	}
	msgs, _ := decodeChatBody(t, out)["messages"].([]any)
	// 形态说明：function_call_output 打断 tool_call 序列 → output 后的第二个
	// function_call 起新开 assistant 消息（配对语义要求 output 紧跟其 call），
	// 故期望 4 条：user / assistant(c1) / tool(c1 output) / assistant(c2)。
	if len(msgs) != 4 {
		t.Fatalf("messages=%d want 4 (user / asst-c1 / tool / asst-c2): %v", len(msgs), msgs)
	}
	asst := msgs[1].(map[string]any)
	tcs, _ := asst["tool_calls"].([]any)
	if len(tcs) != 1 {
		t.Fatalf("asst tool_calls=%d want 1: %v", len(tcs), asst)
	}
	fn0 := tcs[0].(map[string]any)["function"].(map[string]any)
	if fn0["name"] != "multi_agent_v1spawn_agent" {
		t.Errorf("history name=%v want multi_agent_v1spawn_agent", fn0["name"])
	}
	if tcs[0].(map[string]any)["id"] != "call_n1" {
		t.Errorf("history call_id=%v want call_n1（配对键不破）", tcs[0].(map[string]any)["id"])
	}
	if fn0["arguments"] != `{"task":"x"}` {
		t.Errorf("history arguments=%v", fn0["arguments"])
	}
	// namespace=="functions" 不拼接：在第二条 assistant 消息里原名透传。
	asst2 := msgs[3].(map[string]any)
	tcs2, _ := asst2["tool_calls"].([]any)
	if len(tcs2) != 1 {
		t.Fatalf("asst2 tool_calls=%v", asst2)
	}
	fn1 := tcs2[0].(map[string]any)["function"].(map[string]any)
	if fn1["name"] != "plain" {
		t.Errorf("functions namespace must not join: %v", fn1["name"])
	}
	if toolMap == nil {
		t.Fatalf("toolMap must register history namespaced tool")
	}
	if id := toolMap["multi_agent_v1spawn_agent"]; id.Namespace != "multi_agent_v1" || id.Name != "spawn_agent" {
		t.Errorf("toolMap from history=%+v", id)
	}
	// tool 消息（call_id 配对）不受影响。
	toolMsg := msgs[2].(map[string]any)
	if toolMsg["tool_call_id"] != "call_n1" {
		t.Errorf("tool msg=%v", toolMsg)
	}
}

// TestResponsesSyncToResponseNoNamespaceToolMapZeroRegression 无 namespace 工具路径
// 零回归：toolMap 为 nil → function_call name 原样、无 namespace 字段（兼容入口
// responsesSyncToResponse / ResponsesToChat 语义不变）。
func TestResponsesSyncToResponseNoNamespaceToolMapZeroRegression(t *testing.T) {
	// ResponsesToChat 兼容入口：无 namespace 工具时翻译语义与旧签名一致。
	chatBody, err := ResponsesToChat([]byte(`{"model":"m","input":"hi","tools":[{"type":"function","name":"shell","parameters":{"type":"object"}}]}`))
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	if tools, _ := decodeChatBody(t, chatBody)["tools"].([]any); len(tools) != 1 {
		t.Fatalf("tools=%v", tools)
	}
	resp := map[string]any{
		"choices": []any{map[string]any{
			"message": map[string]any{"role": "assistant", "tool_calls": []any{
				map[string]any{"id": "c1", "type": "function",
					"function": map[string]any{"name": "shell", "arguments": "{}"}},
			}},
			"finish_reason": "tool_calls",
		}},
	}
	out := responsesSyncToResponse(resp, "m", responsesReqFields{})
	output, _ := out["output"].([]any)
	fc := output[0].(map[string]any)
	if fc["name"] != "shell" {
		t.Errorf("fc=%v want shell 原样", fc)
	}
	if _, has := fc["namespace"]; has {
		t.Errorf("nil toolMap must not emit namespace: %v", fc)
	}
}

// --- 非流式聚合翻译 ---

// TestResponsesSyncTranslation Aggregate 结果 → Responses 对象。
func TestResponsesSyncTranslation(t *testing.T) {
	chatResp := map[string]any{
		"id":      "chatcmpl-1",
		"object":  "chat.completion",
		"created": int64(1753600000),
		"model":   "glm-5.2",
		"choices": []any{map[string]any{
			"index": 0,
			"message": map[string]any{
				"role":    "assistant",
				"content": "done",
				"tool_calls": []any{map[string]any{
					"id": "call_z", "type": "function",
					"function": map[string]any{"name": "shell", "arguments": "{}"},
				}},
			},
			"finish_reason": "tool_calls",
		}},
		"usage": map[string]any{"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7, "credit": 0.1},
	}
	out := responsesSyncToResponse(chatResp, "glm-5.2", responsesReqFields{})
	raw, _ := json.Marshal(out)
	if out["object"] != "response" || out["status"] != "completed" {
		t.Errorf("object/status=%v %s", out["object"], raw)
	}
	if !strings.HasPrefix(out["id"].(string), "resp_") {
		t.Errorf("id=%v want resp_ prefix", out["id"])
	}
	output, _ := out["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output len=%d want 2 (message+function_call): %s", len(output), raw)
	}
	if output[0].(map[string]any)["type"] != "message" {
		t.Errorf("output[0]=%v", output[0])
	}
	fc := output[1].(map[string]any)
	if fc["type"] != "function_call" || fc["call_id"] != "call_z" || fc["name"] != "shell" {
		t.Errorf("fc=%v", fc)
	}
	usage := out["usage"].(map[string]any)
	// 注意：这里是 Go 字面量 map（数字为 int），经 responsesUsage 原样拷贝；
	// 只有走 JSON 往返后才是 float64。统一归一后比较。
	if numF(t, usage["input_tokens"]) != 3 || numF(t, usage["output_tokens"]) != 4 || numF(t, usage["credit"]) != 0.1 {
		t.Errorf("usage=%v", usage)
	}
}

// numF 数字归一（int/float64 → float64），供字面量 map 断言。
func numF(t *testing.T, v any) float64 {
	t.Helper()
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case int64:
		return float64(n)
	}
	t.Fatalf("not a number: %T %v", v, v)
	return 0
}

// --- 审查修复项回归测试（P0/P1/P2）---

// TestResponsesStreamNoEventsAfterFailed P0-1：error 帧之后到达的任何帧都不得产出事件。
func TestResponsesStreamNoEventsAfterFailed(t *testing.T) {
	frames := "data: {\"error\":{\"message\":\"boom\"}}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"after error\"}}]}\n\n" +
		"data: [DONE]\n\n"
	raw := runTranslator(t, frames, "m")
	evs := respEvents(t, raw)
	for i, e := range evs {
		if e[0] == "response.failed" {
			if i != len(evs)-1 {
				t.Fatalf("events after failed: %v", evs[i+1:])
			}
			return
		}
	}
	t.Fatalf("no failed event: %s", raw)
}

// TestResponsesStreamUsageTotalSynthesized P0-2：末帧 usage 缺 total_tokens → 合成补齐。
func TestResponsesStreamUsageTotalSynthesized(t *testing.T) {
	frames := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}\n\n" +
		"data: [DONE]\n\n"
	raw := runTranslator(t, frames, "m")
	if !strings.Contains(raw, `"total_tokens":10`) {
		t.Errorf("total_tokens must be synthesized (7+3): %s", raw)
	}
}

// TestResponsesSyncUsageTotalSynthesized P0-2 非流式：同口径合成。
func TestResponsesSyncUsageTotalSynthesized(t *testing.T) {
	resp := map[string]any{
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "x"}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 5, "completion_tokens": 2},
	}
	out := responsesSyncToResponse(resp, "m", responsesReqFields{})
	usage := out["usage"].(map[string]any)
	if numF(t, usage["total_tokens"]) != 7 {
		t.Errorf("total_tokens=%v want 7", usage["total_tokens"])
	}
}

// TestResponsesStreamTruncatedToolCallDropped P1-1：length 截断的残缺参数调用不下发，
// completed.status=incomplete；完整参数调用不受影响。
func TestResponsesStreamTruncatedToolCallDropped(t *testing.T) {
	frames := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_ok\",\"type\":\"function\",\"function\":{\"name\":\"good\",\"arguments\":\"{}\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_bad\",\"type\":\"function\",\"function\":{\"name\":\"bad\",\"arguments\":\"{\\\"half\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n" +
		"data: [DONE]\n\n"
	raw := runTranslator(t, frames, "m")
	evs := respEvents(t, raw)
	if strings.Contains(raw, `"call_id":"call_bad"`) {
		t.Errorf("truncated call must not be emitted: %s", raw)
	}
	if !strings.Contains(raw, `"call_id":"call_ok"`) {
		t.Errorf("complete call must survive: %s", raw)
	}
	var comp struct {
		Response struct {
			Status string `json:"status"`
		} `json:"response"`
	}
	if err := json.Unmarshal([]byte(evs[len(evs)-1][1]), &comp); err != nil {
		t.Fatalf("completed data: %v", err)
	}
	if comp.Response.Status != "incomplete" {
		t.Errorf("status=%s want incomplete", comp.Response.Status)
	}
}

// TestResponsesStreamReasoningCapSingleFrame P1-2：单帧超 64KB 的 reasoning 同样封顶。
func TestResponsesStreamReasoningCapSingleFrame(t *testing.T) {
	big := strings.Repeat("r", reasoningCap+1024)
	frames := "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"" + big + "\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	raw := runTranslator(t, frames, "m")
	for _, e := range respEvents(t, raw) {
		if e[0] == "response.reasoning_summary_text.done" {
			var obj map[string]any
			_ = json.Unmarshal([]byte(e[1]), &obj)
			if got := len(obj["text"].(string)); got != reasoningCap {
				t.Errorf("reasoning len=%d want capped %d", got, reasoningCap)
			}
			return
		}
	}
	t.Fatalf("no reasoning_summary_text.done: %s", raw[:200])
}

// TestResponsesToChatEmptyInput400 P1-6：全被跳过的 input（纯 reasoning）→ 400。
func TestResponsesToChatEmptyInput400(t *testing.T) {
	_, err := ResponsesToChat([]byte(`{"model":"m","input":[{"type":"reasoning","summary":[]}]}`))
	if err == nil || !strings.Contains(err.Error(), "no usable messages") {
		t.Fatalf("err=%v want no-usable-messages 400", err)
	}
}

// TestResponsesToChatCustomToolCall P1-6：custom_tool_call / custom_tool_call_output 往返。
func TestResponsesToChatCustomToolCall(t *testing.T) {
	body := `{"model":"m","input":[
		{"type":"custom_tool_call","name":"apply_patch","call_id":"call_c1","input":"*** Begin Patch"},
		{"type":"custom_tool_call_output","call_id":"call_c1","output":"patched"}
	]}`
	out, err := ResponsesToChat([]byte(body))
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	obj := decodeChatBody(t, out)
	msgs, _ := obj["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages=%v", msgs)
	}
	m0 := msgs[0].(map[string]any)
	tcs, _ := m0["tool_calls"].([]any)
	if len(tcs) != 1 {
		t.Fatalf("m0 tool_calls=%v", m0)
	}
	tc := tcs[0].(map[string]any)
	if tc["id"] != "call_c1" {
		t.Errorf("tc id=%v want call_c1", tc["id"])
	}
	fn := tc["function"].(map[string]any)
	if fn["name"] != "apply_patch" || fn["arguments"] != "*** Begin Patch" {
		t.Errorf("fn=%v", fn)
	}
	m1 := msgs[1].(map[string]any)
	if m1["role"] != "tool" || m1["tool_call_id"] != "call_c1" || m1["content"] != "patched" {
		t.Errorf("m1=%v", m1)
	}
}

// TestResponsesToChatCustomTool P1-6：tools 里 custom 形态映射为 function（input 单参数），
// 非 function 服务端工具剔除。
func TestResponsesToChatCustomTool(t *testing.T) {
	body := `{"model":"m","input":"hi","tools":[
		{"type":"custom","name":"apply_patch","description":"freeform"},
		{"type":"web_search","name":"web"},
		{"type":"function","name":"shell","parameters":{"type":"object"}}
	]}`
	out, err := ResponsesToChat([]byte(body))
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	obj := decodeChatBody(t, out)
	tools, _ := obj["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools=%v want 2 (custom→function + function, web_search dropped)", tools)
	}
	customFn := tools[0].(map[string]any)["function"].(map[string]any)
	if customFn["name"] != "apply_patch" {
		t.Errorf("custom tool=%v", customFn)
	}
	params := customFn["parameters"].(map[string]any)
	props := params["properties"].(map[string]any)
	if _, ok := props["input"]; !ok {
		t.Errorf("custom tool parameters must have input property: %v", params)
	}
}

// TestResponsesStreamNestedErrorMessage P1-5：双重编码 message 提升 code/msg。
func TestResponsesStreamNestedErrorMessage(t *testing.T) {
	inner := `{"code":6004,"msg":"model rate limited"}`
	payload, _ := json.Marshal(map[string]any{"error": map[string]any{"message": inner}})
	frames := "data: " + string(payload) + "\n\n"
	raw := runTranslator(t, frames, "m")
	evs := respEvents(t, raw)
	var obj struct {
		Response struct {
			Error struct {
				Code    any    `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		} `json:"response"`
	}
	if err := json.Unmarshal([]byte(evs[len(evs)-1][1]), &obj); err != nil {
		t.Fatalf("failed data: %v", err)
	}
	if numF(t, obj.Response.Error.Code) != 6004 {
		t.Errorf("code=%v want 6004", obj.Response.Error.Code)
	}
	if obj.Response.Error.Message != "model rate limited" {
		t.Errorf("message=%q want inner msg", obj.Response.Error.Message)
	}
}

// TestWriteResponsesErrorCode P1-4：code 透传（不再恒 null）。
func TestWriteResponsesErrorCode(t *testing.T) {
	rec := httptest.NewRecorder()
	writeResponsesError(rec, http.StatusTooManyRequests, "rate_limit_exceeded", "cooling")
	var env struct {
		Error struct {
			Code string `json:"code"`
			Type string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("envelope: %v", err)
	}
	if env.Error.Code != "rate_limit_exceeded" {
		t.Errorf("code=%q want rate_limit_exceeded", env.Error.Code)
	}
	if env.Error.Type != "api_error" {
		t.Errorf("429 type=%s want api_error (only 4xx-local are invalid_request)", env.Error.Type)
	}
}

// TestResponsesSyncEchoRequestFields P2-2：parallel_tool_calls/tool_choice 回显。
func TestResponsesSyncEchoRequestFields(t *testing.T) {
	ptc := false
	resp := map[string]any{
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "x"}, "finish_reason": "stop"}},
	}
	out := responsesSyncToResponse(resp, "m", responsesReqFields{ToolChoice: "required", ParallelToolCalls: &ptc})
	if out["parallel_tool_calls"] != false || out["tool_choice"] != "required" {
		t.Errorf("echo=%v/%v want false/required", out["parallel_tool_calls"], out["tool_choice"])
	}
	// 未携带 → 官方缺省。
	out2 := responsesSyncToResponse(resp, "m", responsesReqFields{})
	if out2["parallel_tool_calls"] != true || out2["tool_choice"] != "auto" {
		t.Errorf("default=%v/%v want true/auto", out2["parallel_tool_calls"], out2["tool_choice"])
	}
}

// TestResponsesStreamReasoningSummaryText P2-4：summary part 携带完整文本，
// reasoning_text.* 旧事件族不再出现。
func TestResponsesStreamReasoningSummaryText(t *testing.T) {
	frames := "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"chain\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n" +
		"data: [DONE]\n\n"
	raw := runTranslator(t, frames, "m")
	if strings.Contains(raw, "response.reasoning_text.") {
		t.Errorf("legacy reasoning_text events must be gone: %s", raw)
	}
	found := false
	for _, e := range respEvents(t, raw) {
		if e[0] == "response.reasoning_summary_text.done" && strings.Contains(e[1], `"text":"chain"`) {
			found = true
		}
	}
	if !found {
		t.Errorf("summary_text.done with full text missing: %s", raw)
	}
}

// TestResponsesSyncReasoningSummary P2-3：非流式 reasoning item 的 summary 携带文本。
func TestResponsesSyncReasoningSummary(t *testing.T) {
	resp := map[string]any{
		"choices": []any{map[string]any{
			"message":       map[string]any{"role": "assistant", "content": "x", "reasoning_content": "think hard"},
			"finish_reason": "stop",
		}},
	}
	out := responsesSyncToResponse(resp, "m", responsesReqFields{})
	output, _ := out["output"].([]any)
	rItem := output[0].(map[string]any)
	summary, _ := rItem["summary"].([]any)
	if len(summary) != 1 || summary[0].(map[string]any)["text"] != "think hard" {
		t.Errorf("summary=%v want think hard", summary)
	}
}

// --- handler 集成（fake upstream 固定 SSE）---

// respSSE 固定成功流：正文 + usage 末帧 + DONE。
const respSSE = "data: {\"id\":\"chatcmpl-9\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"pong\"}}]}\n\n" +
	"data: {\"id\":\"chatcmpl-9\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2,\"credit\":0.0}}\n\n" +
	"data: [DONE]\n\n"

// TestResponsesHandlerStreamIntegration httptest 全链路：/v1/responses 流式 →
// Responses 事件流（复用 handler 测试的 fake upstream 模式）。
func TestResponsesHandlerStreamIntegration(t *testing.T) {
	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		return 200, respSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})
	body := `{"model":"glm-5.2","input":"ping","stream":true}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	ct := rec.Header().Get("Content-Type")
	if !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("content-type=%s want text/event-stream", ct)
	}
	raw := rec.Body.String()
	evs := respEvents(t, raw)
	if evs[0][0] != "response.created" || evs[len(evs)-1][0] != "response.completed" {
		t.Fatalf("first=%s last=%s", evs[0][0], evs[len(evs)-1][0])
	}
	var text strings.Builder
	for _, e := range evs {
		if e[0] == "response.output_text.delta" {
			var obj map[string]any
			_ = json.Unmarshal([]byte(e[1]), &obj)
			text.WriteString(obj["delta"].(string))
		}
	}
	if text.String() != "pong" {
		t.Errorf("text=%q want pong", text.String())
	}
	if !strings.Contains(raw, `"input_tokens":1`) || !strings.Contains(raw, `"output_tokens":1`) {
		t.Errorf("usage mapping missing: %s", raw)
	}
}

// TestResponsesHandlerSyncIntegration 非流式（stream=false）→ Responses JSON 对象。
func TestResponsesHandlerSyncIntegration(t *testing.T) {
	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		return 200, respSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})
	body := `{"model":"glm-5.2","input":"ping","stream":false}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	var obj struct {
		ID     string `json:"id"`
		Object string `json:"object"`
		Status string `json:"status"`
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &obj); err != nil {
		t.Fatalf("response not json: %v body=%s", err, rec.Body)
	}
	if obj.Object != "response" || obj.Status != "completed" || !strings.HasPrefix(obj.ID, "resp_") {
		t.Errorf("envelope=%+v", obj)
	}
	if len(obj.Output) != 1 || obj.Output[0].Type != "message" || obj.Output[0].Content[0].Text != "pong" {
		t.Errorf("output=%+v", obj.Output)
	}
	if obj.Usage.InputTokens != 1 || obj.Usage.OutputTokens != 1 {
		t.Errorf("usage=%+v", obj.Usage)
	}
}

// TestResponsesHandlerBadJSON 畸形请求体 → 400 Responses 错误信封。
func TestResponsesHandlerBadJSON(t *testing.T) {
	var calls int
	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		calls++
		return 200, respSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader("{not json"))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("code=%d want 400", rec.Code)
	}
	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &envelope)
	if envelope.Error.Type != "invalid_request_error" {
		t.Errorf("type=%s want invalid_request_error", envelope.Error.Type)
	}
	if calls != 0 {
		t.Errorf("upstream must not be called, got %d", calls)
	}
}

// TestResponsesHandlerStickyConversation 粘性端到端：conversation 键映射后两次
// 请求命中同一账号（同 fake upstream 单号池恒命中，这里验证 metadata 透传语义）。
func TestResponsesHandlerStickyConversation(t *testing.T) {
	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		return 200, respSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})
	body := `{"model":"glm-5.2","input":"hi","stream":true,"conversation":"conv-e2e"}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	// conversation 键被消费（不报错、正常出流）即证明映射路径无副作用。
	if !strings.Contains(rec.Body.String(), "response.completed") {
		t.Errorf("stream not completed: %s", rec.Body)
	}
}

// TestResponsesHandlerLargeBodyProceeds 无请求体大小上限（max_body_mb 已移除，
// 与 chat 端 TestChatLargeBodyProceeds 同语义）：任意大 body 完整读入并正常进入
// 后续处理（打到上游），网关侧不再 413 预拦截。超限类问题交由上游自然返回错误。
func TestResponsesHandlerLargeBodyProceeds(t *testing.T) {
	var calls int
	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		calls++
		return 200, respSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})
	// 远超旧 8MB 默认上限的合法 JSON body（50MB+）：必须完整读入、照常进上游。
	pad := strings.Repeat("a", 50<<20)
	body := `{"model":"glm-5.2","input":"` + pad + `"}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("code=%d body=%s (oversized body must proceed to upstream, no 413)", rec.Code, rec.Body)
	}
	if calls != 1 {
		t.Errorf("upstream calls=%d want 1 (large body must reach upstream)", calls)
	}
}

// TestResponsesHandlerRotatesOnError 上游 429 → 轮转 + 末端 429（Responses 信封，
// 错误策略与 chat 同语义的直接证据）。
func TestResponsesHandlerRotatesOnError(t *testing.T) {
	var calls int
	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		calls++
		return 200, "data: {\"error\":{\"message\":\"{\\\"code\\\":6004,\\\"msg\\\":\\\"model rate\\\"}\"}}\n\ndata: [DONE]\n\n", true
	})
	p := testPoolWith(
		&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999},
		&auth.Auth{UID: "u2", AccessToken: "at2", ExpiresAt: 9999999999},
	)
	h := NewHandler(Config{Pool: p, Upstream: up})
	body := `{"model":"glm-5.2","input":"hi","stream":true}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	// 6004 error 帧在流内：Responses 翻译器转为 response.failed（流已 200）；
	// 上游侧每次调用都返回同样的失败帧，因此轮转发生在上游调用层（fake 每次 200+error 帧，
	// 不会触发 HTTP 层轮转——这里只验证 handler 不崩、以 failed 事件收尾）。
	if rec.Code != 200 {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	evs := respEvents(t, rec.Body.String())
	if evs[len(evs)-1][0] != "response.failed" {
		t.Errorf("last event=%s want response.failed", evs[len(evs)-1][0])
	}
}
