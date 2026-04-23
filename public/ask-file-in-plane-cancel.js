/**
 * Chainlit UI helpers for this app:
 * 1) File-ask row: add Cancel next to Browse (empty file list = dismiss ask).
 * 2) Autoscroll: keep the thread pinned to the bottom when new steps appear
 *    (Processing…, Indexed…, streamed replies) and when the file picker opens.
 * 3) Header brand: logo + “AI Doc Assistant” in the top bar (uses /public/logo_*.svg via /logo).
 * 4) First paint: branded splash covers Chainlit’s default welcome until the first thread step loads.
 */
(function () {
  function getReactFiber(dom) {
    if (!dom || typeof dom !== "object") return null;
    const key = Object.keys(dom).find(
      (k) => k.startsWith("__reactFiber$") || k.startsWith("__reactInternalInstance$")
    );
    return key ? dom[key] : null;
  }

  function getProps(fiber) {
    if (!fiber) return null;
    return fiber.memoizedProps || fiber.pendingProps || null;
  }

  function findFileAskCallback(fromFiber) {
    let f = fromFiber;
    for (let i = 0; i < 140 && f; i++) {
      const p = getProps(f);
      if (
        p &&
        p.askUser &&
        typeof p.askUser.callback === "function" &&
        p.askUser.spec &&
        p.askUser.spec.type === "file"
      ) {
        return p.askUser.callback;
      }
      f = f.return;
    }
    return null;
  }

  function resolveCancel(browseBtn) {
    const row = browseBtn.closest("div.flex.items-center");
    const fiber =
      getReactFiber(browseBtn) ||
      getReactFiber(browseBtn.parentElement) ||
      (row && getReactFiber(row));
    const cb = findFileAskCallback(fiber);
    if (cb) {
      try {
        cb([]);
      } catch (e) {
        console.warn("[chainlit-ui] cancel ask failed", e);
      }
    }
  }

  function enhanceDropzoneRow() {
    const browse = document.getElementById("ask-upload-button");
    if (!browse || browse.dataset.clDocAssistantCancel === "1") return;

    browse.dataset.clDocAssistantCancel = "1";

    const parent = browse.parentElement;
    if (!parent) return;

    const wrap = document.createElement("div");
    wrap.className = "flex items-center gap-2 ml-auto";
    wrap.setAttribute("data-cl-doc-assistant-ask-actions", "1");

    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.id = "ask-file-cancel-btn";
    cancel.textContent = "Cancel";
    cancel.setAttribute(
      "aria-label",
      "Cancel file upload and return to the chat"
    );
    cancel.addEventListener("click", function (ev) {
      ev.preventDefault();
      ev.stopPropagation();
      resolveCancel(browse);
    });

    parent.insertBefore(wrap, browse);
    browse.classList.remove("ml-auto");
    wrap.appendChild(cancel);
    wrap.appendChild(browse);
  }

  /**
   * Chainlit 2.x: the composer textarea lives in ChatFooter *outside* ScrollContainer
   * once there are messages — walking up from the textarea never reaches the thread
   * scroller. Anchor from #welcome-screen, [data-step-type], or ScrollContainer markup.
   */
  function firstScrollableAncestor(fromEl, minGap) {
    let el = fromEl ? fromEl.parentElement : null;
    for (let i = 0; i < 64 && el; i++) {
      const st = window.getComputedStyle(el);
      const oy = st.overflowY;
      if (
        (oy === "auto" || oy === "scroll") &&
        el.scrollHeight > el.clientHeight + minGap
      ) {
        return el;
      }
      el = el.parentElement;
    }
    return null;
  }

  function pickChainlitInnerScrollPane() {
    const minGap = 6;
    let fallback = null;
    const outers = document.querySelectorAll(
      '[class*="relative"][class*="flex-col"][class*="flex-grow"][class*="overflow-y-auto"]'
    );
    for (let oi = 0; oi < outers.length; oi++) {
      const outer = outers[oi];
      const children = outer.children;
      for (let ci = 0; ci < children.length; ci++) {
        const child = children[ci];
        const cn = child.className && String(child.className);
        if (!cn || !cn.includes("overflow-y-auto")) continue;
        const st = window.getComputedStyle(child);
        if (st.overflowY !== "auto" && st.overflowY !== "scroll") continue;
        if (!fallback && cn.includes("flex-grow")) fallback = child;
        if (child.scrollHeight > child.clientHeight + minGap) {
          return child;
        }
      }
    }
    return fallback;
  }

  function pickChatScrollContainer() {
    const minGap = 6;
    const welcome = document.getElementById("welcome-screen");
    if (welcome) {
      const a = firstScrollableAncestor(welcome, minGap);
      if (a) return a;
    }

    const steps = document.querySelectorAll("[data-step-type]");
    if (steps.length) {
      const a = firstScrollableAncestor(steps[steps.length - 1], minGap);
      if (a) return a;
    }

    const innerPane = pickChainlitInnerScrollPane();
    if (innerPane) return innerPane;

    const textareas = document.querySelectorAll("textarea");
    for (const ta of textareas) {
      const ph = (ta.getAttribute("placeholder") || "").toLowerCase();
      if (!ph) continue;
      if (
        ph.includes("message") ||
        ph.includes("type your") ||
        ph.includes("ask a")
      ) {
        const a = firstScrollableAncestor(ta, minGap);
        if (a) return a;
      }
    }

    let best = null;
    let bestGap = 0;
    document.querySelectorAll('[class*="overflow-y-auto"]').forEach(function (el) {
      const gap = el.scrollHeight - el.clientHeight;
      if (gap > bestGap && gap > 40) {
        bestGap = gap;
        best = el;
      }
    });
    return best;
  }

  function forceScrollChatBottom() {
    const el = pickChatScrollContainer();
    if (el) {
      el.scrollTop = el.scrollHeight;
      return;
    }
    window.scrollTo(0, document.documentElement.scrollHeight);
  }

  /**
   * Pin without scrollIntoView (avoids fighting Chainlit’s inner scroller) and without
   * syncing outer overflow parents (that caused scroll/scroll-button feedback flicker).
   */
  function pinChatScrollPaneIfNearBottom() {
    const el = pickChatScrollContainer();
    if (!el) return;
    const d = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (d <= 1) return;
    if (d > 130) return;
    el.scrollTop = el.scrollHeight;
  }

  function scrollLastThreadNodeIntoView() {
    const steps = document.querySelectorAll("[data-step-type]");
    if (steps.length) {
      steps[steps.length - 1].scrollIntoView({ block: "end", behavior: "auto" });
      return;
    }
    const browse = document.getElementById("ask-upload-button");
    if (browse) {
      browse.scrollIntoView({ block: "end", behavior: "auto" });
      return;
    }
    const welcome = document.getElementById("welcome-screen");
    if (welcome) {
      welcome.scrollIntoView({ block: "end", behavior: "auto" });
    }
  }

  /** Pinned = no scrollIntoView / scrollTop churn (avoids hydration flicker). */
  function scrollContainerDistanceFromBottom() {
    const el = pickChatScrollContainer();
    if (!el) return null;
    return el.scrollHeight - el.scrollTop - el.clientHeight;
  }

  let scrollTimer = null;
  function runScrollToBottomPass() {
    const d = scrollContainerDistanceFromBottom();
    if (d !== null && d <= 2) return;

    scrollLastThreadNodeIntoView();
    requestAnimationFrame(function () {
      requestAnimationFrame(forceScrollChatBottom);
    });
  }

  function scheduleForceScrollChatBottom() {
    if (scrollTimer) clearTimeout(scrollTimer);
    scrollTimer = setTimeout(function () {
      scrollTimer = null;
      runScrollToBottomPass();
    }, 140);
  }

  let welcomeScrollScheduledForEl = null;
  let chatPaneMutationTimer = null;
  let chatPaneObserver = null;

  /**
   * Observe only the thread scroll pane (not #root). Root-level mutations include
   * Chainlit’s floating “scroll down” control mounting/unmounting and cause a pin ↔
   * re-render flicker loop when you sit at the bottom.
   */
  function ensureChatPanePinObserver() {
    const el = pickChatScrollContainer();
    if (!el) return;
    if (el.dataset.clAdaPaneMo === "1") return;
    el.dataset.clAdaPaneMo = "1";

    if (chatPaneObserver) {
      try {
        chatPaneObserver.disconnect();
      } catch (e) {
        /* ignore */
      }
    }

    chatPaneObserver = new MutationObserver(function () {
      if (chatPaneMutationTimer) clearTimeout(chatPaneMutationTimer);
      chatPaneMutationTimer = setTimeout(function () {
        chatPaneMutationTimer = null;
        if (!document.contains(el)) return;
        pinChatScrollPaneIfNearBottom();
      }, 120);
    });
    chatPaneObserver.observe(el, { childList: true, subtree: true });
  }

  /** One delayed pass per welcome mount (avoids burst timers + survives late render). */
  function scheduleInitialWelcomeScrollOnce() {
    const w = document.getElementById("welcome-screen");
    if (!w || welcomeScrollScheduledForEl === w) return;
    welcomeScrollScheduledForEl = w;
    requestAnimationFrame(function () {
      setTimeout(function () {
        if (document.getElementById("welcome-screen") !== w) return;
        runScrollToBottomPass();
      }, 380);
    });
  }

  function scrollFileAskIntoView() {
    const browse = document.getElementById("ask-upload-button");
    const input = document.getElementById("ask-button-input");
    const anchor = browse || input;
    if (!anchor) return;
    anchor.scrollIntoView({ block: "end", behavior: "auto" });
    scheduleForceScrollChatBottom();
  }

  function scrollNearestScrollParentToBottom(fromEl) {
    let el = fromEl;
    for (let i = 0; i < 28 && el; i++) {
      const st = window.getComputedStyle(el);
      const oy = st.overflowY;
      if (
        (oy === "auto" || oy === "scroll") &&
        el.scrollHeight > el.clientHeight + 40
      ) {
        el.scrollTop = el.scrollHeight;
        return true;
      }
      el = el.parentElement;
    }
    return false;
  }

  function ensureHeaderBrand() {
    const header = document.getElementById("header");
    if (!header) return;
    const left = header.querySelector(":scope > div.flex.items-center");
    if (!left) return;

    let brand = document.getElementById("ada-header-brand");
    if (!brand) {
      brand = document.createElement("div");
      brand.id = "ada-header-brand";
      const img = document.createElement("img");
      img.alt = "AI Doc Assistant";
      img.setAttribute("width", "28");
      img.setAttribute("height", "28");
      img.decoding = "async";
      const title = document.createElement("span");
      title.id = "ada-header-app-name";
      title.textContent = "AI Doc Assistant";
      brand.appendChild(img);
      brand.appendChild(title);
      left.insertBefore(brand, left.firstChild);
    }
    const img = brand.querySelector("img");
    /* Header strip is always dark (CSS); set src once to avoid reload flicker. */
    if (img && !img.dataset.adaLogoSet) {
      img.src = "/logo?theme=dark";
      img.dataset.adaLogoSet = "1";
    }
  }

  let fileAskVisible = false;
  function onFileAskVisibilityChange() {
    const has =
      !!document.getElementById("ask-upload-button") ||
      !!document.getElementById("ask-button-input");
    if (has && !fileAskVisible) {
      fileAskVisible = true;
      requestAnimationFrame(scrollFileAskIntoView);
      setTimeout(scrollFileAskIntoView, 160);
    }
    if (!has) fileAskVisible = false;
  }

  function onDomTick() {
    try {
      enhanceDropzoneRow();
    } catch (e) {
      console.warn("[chainlit-ui] dropzone", e);
    }
    try {
      ensureHeaderBrand();
    } catch (e) {
      console.warn("[chainlit-ui] header brand", e);
    }
    onFileAskVisibilityChange();
    scheduleInitialWelcomeScrollOnce();
    ensureChatPanePinObserver();
  }

  let domTickRaf = null;
  function scheduleDomTickFromObserver() {
    if (domTickRaf) return;
    domTickRaf = requestAnimationFrame(function () {
      domTickRaf = null;
      onDomTick();
    });
  }

  const root = document.getElementById("root") || document.documentElement;
  const mo = new MutationObserver(scheduleDomTickFromObserver);
  mo.observe(root, { childList: true, subtree: true });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      onDomTick();
      scheduleInitialWelcomeScrollOnce();
    });
  } else {
    onDomTick();
    scheduleInitialWelcomeScrollOnce();
  }

  document.addEventListener(
    "click",
    function (ev) {
      const btn = ev.target && ev.target.closest("#add-documents-cta");
      if (!btn) return;
      scrollNearestScrollParentToBottom(btn);
      setTimeout(function () {
        scrollNearestScrollParentToBottom(btn);
      }, 60);
      setTimeout(function () {
        scrollNearestScrollParentToBottom(btn);
      }, 200);
      setTimeout(scrollFileAskIntoView, 80);
      setTimeout(scrollFileAskIntoView, 300);
      scheduleForceScrollChatBottom();
    },
    true
  );
})();

/**
 * Covers Chainlit’s default empty-thread screen (Chainlit logo + composer) until the
 * first `data-step-type` message exists—i.e. after `on_chat_start` has sent the welcome
 * markdown and the thread UI replaces WelcomeScreen.
 */
(function () {
  let overlayEl = null;
  let splashObserver = null;
  let failSafeTimer = null;

  function removeThreadSplash() {
    if (failSafeTimer) {
      clearTimeout(failSafeTimer);
      failSafeTimer = null;
    }
    if (overlayEl && overlayEl.parentNode) {
      overlayEl.parentNode.removeChild(overlayEl);
    }
    overlayEl = null;
    if (splashObserver) {
      try {
        splashObserver.disconnect();
      } catch (e) {
        /* ignore */
      }
      splashObserver = null;
    }
  }

  function ensureThreadSplash() {
    if (document.getElementById("ada-thread-splash")) return;
    const welcome = document.getElementById("welcome-screen");
    if (!welcome) return;

    const el = document.createElement("div");
    el.id = "ada-thread-splash";
    el.setAttribute("role", "status");
    el.setAttribute("aria-live", "polite");
    el.setAttribute("aria-label", "Loading AI Doc Assistant");

    const inner = document.createElement("div");
    inner.className = "ada-thread-splash-inner";

    const img = document.createElement("img");
    img.alt = "AI Doc Assistant";
    img.width = 56;
    img.height = 56;
    img.decoding = "async";
    img.src = "/logo?theme=dark";

    const title = document.createElement("div");
    title.className = "ada-thread-splash-title";
    title.textContent = "AI Doc Assistant";

    const sub = document.createElement("div");
    sub.className = "ada-thread-splash-sub";
    sub.textContent = "Connecting to your chat…";

    const spin = document.createElement("div");
    spin.className = "ada-thread-splash-spinner";
    spin.setAttribute("aria-hidden", "true");

    inner.appendChild(img);
    inner.appendChild(title);
    inner.appendChild(sub);
    inner.appendChild(spin);
    el.appendChild(inner);
    document.body.appendChild(el);
    overlayEl = el;
  }

  function syncThreadSplash() {
    const hasSteps = !!document.querySelector("[data-step-type]");
    const welcome = document.getElementById("welcome-screen");

    if (hasSteps || (overlayEl && !welcome)) {
      removeThreadSplash();
      return;
    }
    if (welcome) {
      ensureThreadSplash();
    }
  }

  function scheduleSplashTick() {
    requestAnimationFrame(syncThreadSplash);
  }

  splashObserver = new MutationObserver(scheduleSplashTick);
  const root = document.getElementById("root");
  if (root) {
    splashObserver.observe(root, { childList: true, subtree: true });
  }
  scheduleSplashTick();
  setTimeout(scheduleSplashTick, 40);
  setTimeout(scheduleSplashTick, 200);

  failSafeTimer = setTimeout(function () {
    failSafeTimer = null;
    removeThreadSplash();
  }, 12000);
})();
