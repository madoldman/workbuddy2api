// pipeline.go chat/completions 与 /v1/responses 共用的出站管线：
// 提示词改写、会话粘性、选号轮转、熔断/冷却、成本账本、错误策略与末端错误透传，
// 全部只有这一份实现（两个前端注入协议差异，见 chatPipeSink / responsesPipeSink）。
//
// 提取自原 chatCompletions 内联骨架（零语义变更）；协议差异通过 chatPipeSink
// 的 4 个回调注入，保证既有 chat 行为零回归。
package server

import (
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"workbuddy2api/internal/auth"
	"workbuddy2api/internal/logfmt"
	"workbuddy2api/internal/prompt"
	"workbuddy2api/internal/session"
	"workbuddy2api/internal/upstream"
)

// readChatBody 读取请求体（chat 与 responses 共用）。
// 返回 status=0 表示成功；否则返回错误三元组（status/code/msg），由前端按各自
// 协议信封写出——两协议的 error JSON 形态不同（Responses 客户端也认 OpenAI
// error 信封，但错误码与文案需走各自入口，避免耦合）。
//
// 无大小上限（上游 feat(server)! 移除 server.max_body_mb 预拦截）：body 完整读入，
// 超限类问题交由上游自然返回错误（其响应经既有错误分类链路透出，信息量更大）。
// issue #41 的截断防御语义保留在读错误路径——移除预拦截后，截断只可能来自客户端
// 自己断流，读 body 出错就地 400，不把半截 JSON 喂上游 unmarshal 报 unexpected EOF
// 冤枉罚号。
func (h *Handler) readChatBody(r *http.Request) (body []byte, status int, code, msg string) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, http.StatusBadRequest, "invalid_request", "read body: " + err.Error()
	}
	// 调试开关：设置 WB2A_DUMP_REQ 即把上游侧收到的原始请求体落盘，供离线二分定位指纹命中行。
	// 仅在排查上游指纹拦截时开启；不设置时零开销、不落盘。
	// 只落"大请求"（≥4MB 固定阈值，原 max_body_mb/2 语义的接替）：小探针
	// （{"input":"hi"} 之类）会覆盖掉真正要看的对话请求。
	if os.Getenv("WB2A_DUMP_REQ") != "" && len(body) >= dumpReqMinBytes {
		if err := os.WriteFile("/app/data/last_request.json", body, 0o600); err != nil {
			log.Printf("ERR: [server] dump req: %v", err)
		}
	}
	return body, 0, "", ""
}

// chatPipeSink chat 前端的协议差异注入点。成功路径（SSE 透传 / OpenAI 聚合响应）
// 携带 gateway_hint、chatStat 观测等 chat 特有逻辑；失败路径与 /v1/responses
// 共用（错误策略在共享骨架内统一执行，这里只负责末端信封形态）。
type chatPipeSink struct {
	h           *Handler
	w           http.ResponseWriter
	bareModel   string
	reqHasImage bool
	start       time.Time // TTFB 计时起点（st.start）
}

func (s chatPipeSink) writer() http.ResponseWriter { return s.w }

func (s chatPipeSink) writeEndError(w http.ResponseWriter, status int, code, msg, hint string) {
	writeOpenAIErrorHint(w, status, code, msg, hint)
}

func (s chatPipeSink) writeImmediateError(w http.ResponseWriter, status int, code, msg, hint string) {
	writeOpenAIErrorHint(w, status, code, msg, hint)
}

func (s chatPipeSink) onStreamSuccess(w http.ResponseWriter, h *Handler, acct *auth.Auth, rc io.ReadCloser, st *chatStat, bareModel string, ctxFn func() upstream.HintContext) bool {
	// 流式：透传结束后立即关闭上游 body，避免 defer 在轮转场景下堆积 fd。
	stats := newChatStatsReaderSince(rc, s.start)
	// gateway_hint（SSE）：成功状态 200 已开流，中途 error 帧透传时附加
	// hint 字段（hintFn 惰性求值——正常流零开销，只有真撞到 error 帧才
	// 组装请求上下文做判定）。
	sErr := upstream.StreamHint(w, stats, upstream.FrameHintFunc(ctxFn))
	if upstream.IsEmptyStreamError(sErr) {
		// 上游 200 但空流（0 有效帧）：StreamHint 已写 error 帧 + [DONE]
		// 兜底（HTTP 头已发出只能 200），但这是上游缺陷不是成功——日志/
		// 状态收敛到 502 观测，与非流式 Aggregate 空流→502 upstream_parse
		// 同语义（此前 `_ =` 吞错把失败流记成 200，运维看到假成功）。
		// 只认 IsEmptyStreamError：客户端断连的写失败不误标（人已走，
		// 502 观测没有意义）。
		st.status = http.StatusBadGateway
		log.Printf("WARN: [server] stream acct=%s model=%s: empty upstream stream (200+0 frames)", logfmt.Label(acct.UID, acct.Nickname), bareModel)
	}
	st.ttfb = stats.TTFB()
	// usage 缺失时保留 chatStat.toks 的 -1 哨兵（观测缺失 → 显示 "-"），
	// 不写入零值——否则「没观测到 usage」被伪造成「测得 0 token」，
	// 与非流式走 completionTokens 返回 -1 的口径不一致。
	toks, hasUsage := stats.Tokens()
	if hasUsage {
		st.toks = toks
	}
	// metrics 采集：token 三段 + 缓存三段 + 真实扣费（供 /v1/stats）。
	// 与成本账本同源同口径（都读末帧 usage），故此处一并带出，避免二次解析。
	st.hasUsage = hasUsage
	st.prompt = stats.PromptTokens()
	st.cacheHit, st.cacheMiss, st.cacheWr = stats.CacheTokens()
	if credit, ok := stats.Credit(); ok {
		st.credit = credit
		st.hasCredit = true
	}
	// 成本账本：末帧 usage 带 credit 与 token 总数时记录实测单价，
	// 供下次选号把免费/便宜的号排在前面。
	if credit, ok := stats.Credit(); ok {
		h.cfg.Pool.NoteModelCost(acct.UID, bareModel, credit, stats.TotalTokens())
	} else if hasUsage {
		// R9(c) 防护观测：usage 存在但 credit 缺失（如 global SSE 末帧未带 credit）。
		// 不算合法成本观测（缺失≠0），仅记一条 WARN 协助排障，绝不写入账本。
		log.Printf("WARN: [server] stream usage without credit acct=%s model=%s (no cost observation)", logfmt.Label(acct.UID, acct.Nickname), bareModel)
	}
	rc.Close()
	return true
}

func (s chatPipeSink) onSyncSuccess(w http.ResponseWriter, h *Handler, acct *auth.Auth, rc io.ReadCloser, st *chatStat, bareModel string) {
	resp, err := upstream.Aggregate(rc)
	rc.Close()
	if err != nil {
		// 上游流解析失败：客户端还没看到任何输出，回 502 并告知原因。
		writeJSON(w, http.StatusBadGateway, map[string]any{
			"error": map[string]any{
				"message": err.Error(),
				"type":    "api_error",
				"code":    "upstream_parse",
			},
		})
		st.status = http.StatusBadGateway
		return
	}
	writeJSON(w, http.StatusOK, resp)
	st.status = http.StatusOK
	st.toks = completionTokens(resp)
	// 成本账本（非流式）：从聚合响应的 usage 取 credit 与 token 总数。
	if credit, total, ok := usageCreditTotal(resp); ok {
		h.cfg.Pool.NoteModelCost(acct.UID, bareModel, credit, total)
	}
	// metrics 采集（非流式）：与流式同口径，从同一份 usage 带出。
	fillStatFromUsage(st, resp)
}

// runChatPipeline 共享轮转骨架。从原 chatCompletions 内联代码提取（注释保留原样），
// sink 注入两个前端的协议差异。st 由调用方经 newChatStat 构建后传入（chat/responses
// 两个前端均如此），骨架内直接解引用不做 nil 防御——观测写入是骨架职责的一部分。
// pipeSink 协议前端的差异注入接口：chat 与 responses 各自实现成功路径的
// 响应写出（错误路径两端形态足够接近，由 writeEndError/writeImmediateError 注入）。
type pipeSink interface {
	writer() http.ResponseWriter
	writeEndError(w http.ResponseWriter, status int, code, msg, hint string)
	writeImmediateError(w http.ResponseWriter, status int, code, msg, hint string)
	onStreamSuccess(w http.ResponseWriter, h *Handler, acct *auth.Auth, rc io.ReadCloser, st *chatStat, bareModel string, ctxFn func() upstream.HintContext) bool
	onSyncSuccess(w http.ResponseWriter, h *Handler, acct *auth.Auth, rc io.ReadCloser, st *chatStat, bareModel string)
}

func (h *Handler) runChatPipeline(r *http.Request, body []byte, peekModel, bareModel, realm string, stream bool, reqHasImage bool, st *chatStat, sink pipeSink) {
	w := sink.writer()
	tried := map[string]bool{}
	var lastErr error
	// 会话粘性：从请求体提取会话键并解析绑定号（找不到/无效则 stickyUID 为空，走普通轮换）。
	// 按模型解析：同一个会话可能换模型，绑定号若在当前模型上被 6004 限额（对其他模型
	// 仍可用），必须重分配——否则会被钉在这个号上反复失败。
	// 提取与下方会话头族的聚合键共用同一结果，故**不受粘性开关影响**：粘性未启用
	// （Session==nil）时聚合键仍应是会话级，而不是退化成轮级。
	sessKey := session.ExtractKey(body)
	// stickyKey 是**粘性专用**键，与 sessKey（会话头族聚合用）分开：
	// sessKey 为空时（OpenAI 兼容客户端——dsh / Codex 等既无 conversationId 也无
	// metadata）用首条 user 消息派生会话级 fallback 键，使粘性仍能生效。
	// 不能直接改 sessKey：那会连带改变上游头族轮级复合键（sessKey 入键）的聚合
	// 语义，属于另一条链路的契约。
	stickyKey := sessKey
	if stickyKey == "" {
		stickyKey = session.StickyFallbackKey(body)
	}
	stickyUID := ""
	if h.cfg.Session != nil && stickyKey != "" {
		// 传给 ResolveForModel 的是**完整**模型名（peek.Model，含 realm 前缀）。
		// 粘性命中校验走 injected AvailableForModel 闭包 → 闭包内部 resolveModel 剥前缀
		// 得 realm+bare，再按 realm 过滤可用集合。若传已剥前缀的 bareModel，闭包对裸名
		// 恒剥出 realm=cn，跨 realm 粘性会话会被错误钉回 CN 集合；完整前缀才能让
		// 闭包正确过滤到 global 集合（见 cmd/server/wiring.go realmAwareAvailableForModel）。
		// 模型名也参与成本账本与选号过滤，不能用 "-" 占位污染模型键。
		if uid, ok := h.cfg.Session.ResolveForModel(stickyKey, peekModel); ok {
			stickyUID = uid
		}
	}

	// 轮级聚合键：按 body 里最后一条 user 消息派生（同轮内所有上游调用同键，
	// 换 user 消息换键）。#170 起带会话键的客户端也统一走轮级（与官方桌面 CLI 的
	// X-Conversation-Request-ID 轮级语义对齐），故不再限 sessKey=="" 才计算；
	// sessKey 由调用侧以复合键方式入键（防不同会话同轮文本互撞）。
	// 必须在下方 prompt.Rewrite / rewriteModel 之前取——改写会动 messages 内容。
	turnKey := session.TurnKey(body)

	// 在途租约：成功选中即占名额；函数出口（含成功 return 与 panic）统一释放。
	var heldUID string
	defer func() {
		if heldUID != "" {
			h.cfg.Pool.Release(heldUID)
		}
	}()
	releaseHeld := func() {
		if heldUID != "" {
			h.cfg.Pool.Release(heldUID)
			heldUID = ""
		}
	}
	// unbindSticky 解绑当前会话粘性号（stickyUID 非空时）。供「粘性号不可用/被抢」与 fail 共用。
	// 幂等：stickyUID 已空则空操作；不会误解绑其他轮的绑定。仅当 Session != nil 时 stickyUID 才会非空。
	unbindSticky := func() {
		if stickyUID != "" {
			h.cfg.Session.Unbind(stickyKey)
			stickyUID = ""
		}
	}
	// fail 在轮转失败分支统一：释放租约 + 若失败号正是粘性号则解绑（下次请求重新分配）。
	fail := func(uid string) {
		releaseHeld()
		if stickyUID != "" && uid == stickyUID {
			unbindSticky()
		}
	}

	// 系统提示词改写（出站前、轮转前；每个请求一次）。
	//   - custom：用自有提示词替换客户端 system/developer（从源头消灭 system 指纹误报）。
	//   - append：开头连续 system/developer 块后插自有提示词，既有消息逐字不动
	//     （客户端项目规范/工具约定与网关提示词并用，issue #129）。
	//   - passthrough + 降级期：换 Degraded 中性提示词直达，不再先撞 400。
	//   - passthrough / append 非降级期：透传客户端原始 system（append 则再插一条网关 system）。
	// 降级裁决：append 在降级期退化为 replace（Rewrite(Degraded)）——append 带
	// 指纹原文重试是确定性再撞墙，replace 是一次性最小抢救（issue #129 设计 §4）。
	degradedApplied := false
	if h.cfg.PromptMode == "custom" && h.cfg.PromptText != "" {
		body = prompt.Rewrite(body, h.cfg.PromptText)
	} else if h.cfg.PromptMode == "append" && h.cfg.PromptText != "" && !h.degrade.Active() {
		body = prompt.Append(body, h.cfg.PromptText)
	} else if (h.cfg.PromptMode == "passthrough" || h.cfg.PromptMode == "append") && h.degrade.Active() {
		body = prompt.Rewrite(body, prompt.Degraded)
		degradedApplied = true
	}

	// outbound model 名重写为 bareModel（D6）：realm 前缀是网关侧路由协议，
	// 上游不认前缀（global 账号也请求裸模型名）。裸名时 bareModel==peek.Model 恒等。
	if bareModel != peekModel {
		body = rewriteModel(body, bareModel)
	}

	// 会话头族（issue #35）：后台按 X-Conversation-Request-ID 聚合请求，官方客户端
	// 一次 user send 内所有 tool call/重试/换号复用同一个 ID。**双形态并存**（均对齐
	// 官方）：带 conversationId 的客户端为**会话级**（对齐官方云链路 lfConvReqId 的
	// 服务端下发后复用形态，跨轮同键）；无会话键的客户端为**轮级**（对齐官方桌面
	// CLI / 排队链路的 queueRequestId 形态，同轮内复用、换 user 消息换键）。此处
	// **轮转循环外**生成一次，循环内每次出站原样复用 → 换号/重试/降级全部同 ID，
	// 后台不再碎片化（此前网关一个都不发，上游按 HTTP 请求逐条记账，同一对话几十
	// 上百个 RequestID）。
	//   - conversationID：body 提取（透传客户端原值，缺省空串——不伪造，见
	//     ResolveConversationID；官方后台不校验一致，空会话则不建立聚合键）；
	//   - conversationRequestID：入站 X-Conversation-Request-ID 透传优先（客户端已
	//     有自己的对话轮 ID 则以客户端为准），否则按粘性 key 进程内稳定生成（同会话
	//     恒同值）；粘性 key 也为空时走轮级兜底（session.TurnKey/TurnRequestID），
	//     无 user 消息时退化成本请求级 NewMessageID——轮转内捕获一次即共享；
	//   - messageID 在 ChatHeaders 内每条消息生成（消息级独立，无需外部可见）。
	// 会话头族（issue #35 / #170）：后台按 X-Conversation-Request-ID 聚合请求，官方
	// 客户端一次 user send 内所有 tool call/重试/换号复用同一个 ID。**统一轮级**
	// （对齐官方桌面 CLI：TraceStartHook 每次 USER_PROMPT_SUBMIT 清空重生成
	// conversationRequestId，同轮内复用、跨轮必换；官方云链路 lfConvReqId 的会话级
	// 是服务端指令，走本网关的客户端不属于该形态）。此处**轮转循环外**生成一次，
	// 循环内每次出站原样复用 → 换号/重试/降级全部同 ID，后台不再碎片化（此前网关
	// 一个都不发，上游按 HTTP 请求逐条记账，同一对话几十上百个 RequestID）。
	//   - conversationID：body 提取（透传客户端原值，缺省空串——不伪造，见
	//     ResolveConversationID；官方后台不校验一致，空会话则不建立聚合键）；
	//   - conversationRequestID：入站 X-Conversation-Request-ID 透传优先（客户端已
	//     有自己的对话轮 ID 则以客户端为准）。派生分两态：
	//     * turnKey 非空（有末条 user 消息）→ 带会话键客户端走 TurnRequestID(
	//       sessKey+":"+turnKey) 复合键（会话段入键保证不同会话同轮文本不互撞，
	//       轮级粒度对齐官方 CLI）；无会话键客户端走既有 TurnRequestID(turnKey)
	//       纯轮级键（存量会话键值零漂移）。
	//     * turnKey 为空（残留空态：无 user 消息/无可签名内容）→ sessKey 非空时
	//       回落 RequestIDForKey(sessKey)（会话级兜底，好于请求级随机）；sessKey
	//       也空走 NewMessageID 请求级（TurnRequestID 空键行为）——轮转内捕获
	//       一次即共享。
	//   - messageID 在 ChatHeaders 内每条消息生成（消息级独立，无需外部可见）。
	chatMeta := upstream.ChatMeta{ConversationID: session.ResolveConversationID(body)}
	if v := r.Header.Get("X-Conversation-Request-ID"); v != "" {
		chatMeta.ConversationRequestID = v
	} else if turnKey != "" && sessKey != "" {
		// 轮级复合键：sessKey 入键防跨会话同轮文本互撞。
		chatMeta.ConversationRequestID = session.TurnRequestID(sessKey + ":" + turnKey)
	} else if turnKey != "" {
		// 无会话键客户端：纯轮级键（既有兜底语义不变，存量会话键值零漂移）。
		chatMeta.ConversationRequestID = session.TurnRequestID(turnKey)
	} else if sessKey != "" {
		// 残留空态兜底：无轮可聚合时维持会话级聚合（同会话恒同值）。
		chatMeta.ConversationRequestID = session.RequestIDForKey(sessKey)
	} else {
		// 无会话键也无轮级键：请求级随机（轮转内捕获一次即共享）。
		chatMeta.ConversationRequestID = session.TurnRequestID("")
	}
	chatMeta.TraceID = r.Header.Get("X-Trace-ID")

	for i := 0; i < h.cfg.MaxRotate; i++ {
		// 选号：粘性号优先（PickByUIDForModel 已校验该模型可用性 + 在途未满），否则普通轮换。
		var acct *auth.Auth
		if stickyUID != "" {
			acct = h.cfg.Pool.PickByUIDForModel(stickyUID, bareModel)
			if acct == nil || (realm != "" && acct.Realm() != realm) {
				// 粘性号在当前模型不可用（冷却/占满/该模型被 6004 限额）或 realm 不符 → 解绑。
				unbindSticky()
			}
		}
		if acct == nil {
			// 模型感知 + realm 感知选号：模型非空时启用 6004 模型级冷却豁免
			// （healthyForModel），realm 谓词过滤跨域账号。
			acct = h.cfg.Pool.PickExcludingForRealm(tried, bareModel, realm)
		}
		if acct == nil {
			st.status = http.StatusServiceUnavailable
			break
		}
		st.uid = acct.UID
		// 同步昵称：请求流水行只写 uid8 时无法直观看是哪一号，昵称随本次选号带入日志行。
		st.nick = acct.Nickname
		tried[acct.UID] = true

		// 占用在途名额：Pick 已跳过满额账号，此处 CAS 兜底并发抢名额的竞态。
		if !h.cfg.Pool.Acquire(acct.UID) {
			// 若被抢的正是粘性号，立即解绑并回落普通轮换，避免下一轮仍撞同一个
			// 满载粘性号再浪费一次粘性命中往返（语义与 fail()/粘性命中-nil 的解绑一致）。
			if stickyUID != "" && acct.UID == stickyUID {
				unbindSticky()
			}
			if !rotateBackoff(i, r.Context()) {
				// 客户端已断连：换号重试无意义，终止轮转走末端错误透传。
				break
			}
			continue // 最后一个名额被并发抢走 → 换号
		}
		heldUID = acct.UID

		// token 临近过期 → 先 refresh（失败冷却换号）
		if acct.NeedsRefresh(h.cfg.RefreshSkew) {
			if err := h.cfg.Upstream.RefreshToken(acct); err != nil {
				lastErr = err
				var ue *upstream.Error
				if errors.As(err, &ue) && ue.Kind == upstream.ErrSessionDead {
					// 12153 一次失败不杀号（临时触发会误杀）：与 scheduler keepalive/checkin
					// 同口径走连续计数，达到 sessionDeadThreshold 才禁用。
					h.cfg.Pool.NoteSessionDead(acct.UID)
				} else {
					h.cfg.Pool.NoteError(acct.UID)
				}
				fail(acct.UID)
				if !rotateBackoff(i, r.Context()) {
					break // ctx 取消：终止轮转（refresh 失败换号退避，WAF P0-2）
				}
				continue
			}
			acct.BackfillRealm() // 老 auth 空 realm → 落盘前补标识（幂等：已有不动）
			if err := acct.SaveAtomic(); err != nil {
				// 刷新成功但落盘失败：下次启动会用旧 token，必须暴露
				log.Printf("ERR: [server] chat refresh acct=%s: save auth failed: %v", logfmt.Label(acct.UID, acct.Nickname), err)
			}
		}

		// 客户端 IP 透传（仅 PassthroughIP 开启）：按请求取首段作为参数传入 ChatStream，
		// 不再读写共享字段——并发请求各自携带独立 IP，互不串扰（issue：ClientIP 竞态）。
		var clientIP string
		if h.cfg.Upstream.PassthroughIP {
			clientIP = upstream.ExtractClientIP(r)
		}
		// 传 r.Context()：客户端断连/请求取消立即中断在途上游调用并释放租约，
		// 不再让"幽灵请求"占满账号在途名额直到 IdleTimeout。
		rc, status, respBody, terr := h.cfg.Upstream.ChatStreamContext(r.Context(), acct, body, clientIP, chatMeta)
		// 分类信封一次成型：upstream 已在错误路径返回 *upstream.Error（Kind +
		// Retry-After 头解析，见 ChatStreamContext 注释）。传输层错误（非 *Error）走
		// 抖动换号分支；防御分支（terr 为 nil 但 status>=400，如 ErrNone 兜底）回落
		// 本地 Classify，双保险不改变语义。
		var uerr *upstream.Error
		if errors.As(terr, &uerr) {
			status = uerr.Status
		}
		if uerr == nil && terr != nil {
			// 网络层抖动：只换号，不喂熔断计数（传输层错误对连续失败连坐熔断过于严苛）。
			// 连败兜底（issue #114）：喂连败计数——连不上上游是「不知道原因的失败」，
			// 连败 N 次临时出池，单次/偶发不罚（NoteFailures 内部达阈才动作）。
			// 上游 client 已打 transport error 日志。
			st.status = http.StatusServiceUnavailable
			lastErr = terr
			h.cfg.Pool.NoteFailures(acct.UID)
			fail(acct.UID)
			if !rotateBackoff(i, r.Context()) {
				break // ctx 取消：终止轮转（传输层错误换号退避，WAF P0-2）
			}
			continue
		}
		if status >= 400 {
			st.status = status
			var kind upstream.ErrKind
			if uerr != nil {
				kind = uerr.Kind
			} else {
				kind = upstream.Classify(status, string(respBody))
				uerr = &upstream.Error{Kind: kind, Status: status, Msg: string(respBody)}
			}
			// 内容拦截误报（passthrough/append 模式首遇）：判定为 system 指纹误报，
			// 触发降级到次日 00:00 CST，换 Degraded 中性提示词同请求内重试（append
			// 降级重试同样退化为 replace——原文在场只会确定性再撞 400）。
			// 第二次仍被拦（用户内容本身触发审核）→ 回内容防火墙错误（见下分支）。
			// 内容问题非账号问题：applyErrorPolicy 不罚账号（见 ErrContentBlocked 分支）。
			if kind == upstream.ErrContentBlocked && (h.cfg.PromptMode == "passthrough" || h.cfg.PromptMode == "append") && !degradedApplied {
				h.degrade.Trigger()
				body = prompt.Rewrite(body, prompt.Degraded)
				degradedApplied = true
				delete(tried, acct.UID) // 单账号池也能拿到重试机会（降级重试占一次名额）
				releaseHeld()
				log.Printf("WARN: [server] content-blocked (likely fingerprint false positive) -> degraded prompt retry")
				continue
			}
			if kind == upstream.ErrContentBlocked {
				// 内容命中网关内容防火墙：立即回客户端，**不轮转**——换任何账号都会撞同一
				// 审核，轮转纯属浪费时间。不罚账号（ErrContentBlocked 分支无冷却/熔断/NoteError）。
				// error-passthrough：message 装上游 body 原文（code/msg/requestId 原样，
				// 任务书授权上游错误码/账号语义对客户端可见），不再改写成网关固定文案。
				h.applyErrorPolicy(acct.UID, kind, string(respBody), bareModel, uerr)
				fail(acct.UID)
				msg := string(respBody)
				if strings.TrimSpace(msg) == "" {
					// 空 body 兜底：无上游原文可透传，保留可读分类文案（不编造原文）。
					msg = "content blocked by upstream content firewall"
				}
				sink.writeImmediateError(w, http.StatusBadRequest, "content_blocked", msg,
					h.hintOf(upstream.ErrContentBlocked, string(respBody), bareModel, reqHasImage, uerr))
				st.status = http.StatusBadRequest
				return
			}
			// 11115「prompt is too long」：立即透传上游原文回客户端，**不罚号不轮转**
			// ——上下文超限是请求的问题（同一 body 换任何号都超限，白扔健康号配额；
			// 与 WAF IP fail-fast 同哲学：确定与账号无关的错误直接终止轮转）。
			// applyErrorPolicy ErrPromptTooLong 分支零动作（不冷却/不熔断/不 NoteError，
			// 不喂连败），fail 只释放租约。error-passthrough：message 装上游 body 原文
			// （code/msg/requestId 原样，含真实 token 数与上限值——上游原文是最有价值
			// 的错误信息，客户端必须看到，禁止固定词覆盖）。
			if kind == upstream.ErrPromptTooLong {
				h.applyErrorPolicy(acct.UID, kind, string(respBody), bareModel, uerr)
				fail(acct.UID)
				sink.writeImmediateError(w, http.StatusBadRequest, "prompt_too_long", promptTooLongMessage(string(respBody)),
					h.hintOf(upstream.ErrPromptTooLong, string(respBody), bareModel, reqHasImage, uerr))
				st.status = http.StatusBadRequest
				return
			}
			// 图片格式/数据无效：立即透传上游原文回客户端，不罚号不轮转。
			// 同一 body 换账号仍是同样的解析结果，轮转只会放大无效请求。
			if kind == upstream.ErrImageInvalid {
				h.applyErrorPolicy(acct.UID, kind, string(respBody), bareModel, uerr)
				fail(acct.UID)
				msg := string(respBody)
				if strings.TrimSpace(msg) == "" {
					msg = "image request was rejected by upstream"
				}
				sink.writeImmediateError(w, http.StatusBadRequest, "image_invalid", msg,
					h.hintOf(upstream.ErrImageInvalid, string(respBody), bareModel, reqHasImage, uerr))
				st.status = http.StatusBadRequest
				return
			}
			// lastErr 携带完整 body（uerr.Msg 在 upstream 侧截断 200 字符，透传语义
			// 5755fe3 要求原文全量）+ Kind/RetryAfter（末端映射与冷却时长共用）。
			lastErr = &upstream.Error{Kind: kind, Status: status, Msg: string(respBody), RetryAfter: uerr.RetryAfter}
			h.applyErrorPolicy(acct.UID, kind, string(respBody), bareModel, uerr)
			fail(acct.UID)
			// WAF IP 级 fail-fast（优先于 rotateBackoff 退避——IP 级拦截时退避无意义，
			// 任务书设计纪律）：该次 WAF 403 喂入 IP 级状态机，若激活（短窗多号命中，
			// IP 被拦而非账号）则立即终止轮转——继续换号只会把请求放大 MaxRotate 倍
			// 打同一出口 IP，加重风控。账号级软冷却已在上方 applyErrorPolicy 照常记账
			// （单号偶发 403 仍冷却），IP 级状态只改变「是否继续轮转」——协同不叠加。
			if kind == upstream.ErrWafBlock && h.wafIP.noteWaf(acct.UID) {
				break
			}
			if !rotateBackoff(i, r.Context()) {
				break // ctx 取消：终止轮转（分类错误换号退避，WAF P0-2）
			}
			continue
		}
		h.cfg.Pool.NoteSuccess(acct.UID)
		// 11102 负缓存清命：该账号该模型实测成功，立即解除避让（不必等 TTL 到期）。
		// BlockModelClear 按 "11102" reason 前缀识别，只清 11102 条目、不碰 6004 独立冷却。
		h.cfg.Pool.BlockModelClear(acct.UID, bareModel)
		// 粘性跟随最终成功号：本轮成功的账号成为该会话的粘性绑定（覆盖旧绑定）。
		// 若 sticky 号失败、轮换到别的号成功，这里把会话重绑到新号，多轮对话下一跳不再随机抽。
		if stickyKey != "" && h.cfg.Session != nil {
			h.cfg.Session.Bind(stickyKey, acct.UID)
		}
		if stream {
			st.status = http.StatusOK
			sink.onStreamSuccess(w, h, acct, rc, st, bareModel, func() upstream.HintContext {
				return h.hintContext(bareModel, reqHasImage)
			})
			return
		}
		sink.onSyncSuccess(w, h, acct, rc, st, bareModel)
		return
	}
	// 末端错误透传（error-passthrough）：上游返回的错误原样透传，不再规范化成固定文案。
	//
	// 背景：此前把上游原始错误文本统一改写，防账号 UID / 上游内部错误码（11128 / 12153 /
	// 6004 后台措辞）泄露——但副作用是客户端看不到真实错误，根本没法排查上游问题。
	// 任务书规定上游错误码/账号语义**允许**泄露给客户端（有意为之），排查必须看到原文。
	//
	//   - 上游返回（*upstream.Error）→ error.message 装**上游 body 原文**（code/msg/
	//     requestId 原样保留，如 {"code":6004,"msg":"…","requestId":"…"}）。HTTP 状态码
	//     按 OpenAI 兼容口径映射类别：ErrSoftRate → 429（限流语义、客户端应等待重试），
	//     其余保持 503（网关侧无健康账号可用）。业务 code 取本地分类可读名
	//     （rate_limit_exceeded / no_healthy_account）。
	//   - 本地调度类错误（无可用账号 acct==nil、传输层抖动、非上游返回的 lastErr）
	//     → 保留自有文案 no_healthy_account（本地错误没有上游原文可透传，不编造）。
	status := http.StatusServiceUnavailable
	code := "no_healthy_account"
	msg := "all accounts are temporarily unavailable, please retry later"
	// gateway_hint（末端透传）：上游错误按 Kind + 原文 + 请求形态判定（11133/11135
	// 在 hint 层自带形态判定，ErrClient 家族也能带上 hint）；本地调度类错误
	// （无上游原文）固定 no_healthy_account hint。
	hint := upstream.NoHealthyAccountHint()
	var ue *upstream.Error
	if errors.As(lastErr, &ue) {
		hint = h.hintOf(ue.Kind, ue.Msg, bareModel, reqHasImage, ue)
		switch ue.Kind {
		case upstream.ErrSoftRate:
			status = http.StatusTooManyRequests
			code = "rate_limit_exceeded"
			msg = "rate limited: all accounts are cooling down, please wait a moment and try again"
		case upstream.ErrWafBlock:
			if h.wafIP.active() {
				// IP 级拦截措辞（fail-fast 终止路径）：空 body 时给出明确可读文案——
				// 网关出口 IP 被 WAF 拦截、轮转已止损、窗口 X 秒后自动解除。客户端
				// 提前重试无意义（换号不换 IP）；有上游原文时原文优先（下方统一）。
				code = "waf_ip_blocked"
				msg = "waf ip-level block: upstream firewall is blocking the gateway IP, rotation stopped; retry after the block window expires"
			}
		}
		if s := strings.TrimSpace(ue.Msg); s != "" {
			// 上游原文优先：透传 code/msg/requestId，不拼接本地前缀。
			msg = s
		}
	}
	sink.writeEndError(w, status, code, msg, hint)
	st.status = status
}
