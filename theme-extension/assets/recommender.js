/* khriss recommender — client widget.
 *
 * Flow: shopper picks an image -> show preview -> POST to {api}/recommend ->
 * render a results grid. Mode A ("outfit") shows a stylist rationale under each
 * card; Mode B ("shoe") shows a plain similarity grid; ambiguous ("both")
 * renders two tabs backed by the two result lists in the response.
 *
 * All copy comes from data-khriss-t-* attributes set in the Liquid block, so
 * this file carries no user-facing strings and stays translatable.
 */
(function () {
  const roots = document.querySelectorAll(".khriss");
  roots.forEach(initWidget);

  function initWidget(root) {
    const api = (root.dataset.khrissApi || "").replace(/\/$/, "");
    const t = readStrings(root);
    const money = root.dataset.khrissMoney || "{{amount}}";

    const fileInput = root.querySelector(".khriss__file");
    const drop = root.querySelector(".khriss__drop");
    const preview = root.querySelector(".khriss__preview");
    const previewImg = root.querySelector(".khriss__preview-img");
    const status = root.querySelector(".khriss__status");
    const tabs = root.querySelector(".khriss__tabs");
    const resultsEl = root.querySelector(".khriss__results");

    let lastResponse = null;
    let pending = false;

    fileInput.addEventListener("change", () => {
      const file = fileInput.files && fileInput.files[0];
      if (file) handleFile(file);
    });

    // Drag-and-drop onto the upload area, plus keyboard activation.
    ["dragenter", "dragover"].forEach((ev) =>
      drop.addEventListener(ev, (e) => {
        e.preventDefault();
        drop.classList.add("is-dragging");
      })
    );
    ["dragleave", "drop"].forEach((ev) =>
      drop.addEventListener(ev, (e) => {
        e.preventDefault();
        drop.classList.remove("is-dragging");
      })
    );
    drop.addEventListener("drop", (e) => {
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) handleFile(file);
    });
    drop.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    async function handleFile(file) {
      if (pending) return; // ignore a second pick while one is in flight
      pending = true;

      previewImg.src = URL.createObjectURL(file);
      preview.hidden = false;
      lastResponse = null;
      setTabsVisible(false);
      setStatus(t.analysing, { busy: true });
      showSkeletons();

      try {
        const data = await postImage(api, file);
        lastResponse = data;
        render(data);
      } catch (err) {
        resultsEl.innerHTML = "";
        setStatus(t.error);
        console.error("khriss:", err);
      } finally {
        pending = false;
      }
    }

    tabs.querySelectorAll(".khriss__tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        // Guard: the tabs can be reached before any response has landed.
        if (!lastResponse) return;
        tabs.querySelectorAll(".khriss__tab").forEach((b) => {
          b.classList.remove("is-active");
          b.setAttribute("aria-selected", "false");
        });
        btn.classList.add("is-active");
        btn.setAttribute("aria-selected", "true");
        const which = btn.dataset.tab;
        renderGrid(
          which === "outfit" ? lastResponse.outfit_results : lastResponse.shoe_results,
          which
        );
      });
    });

    function render(data) {
      if (data.mode === "both") {
        setStatus(t.both);
        setTabsVisible(true);
        renderGrid(data.outfit_results || [], "outfit");
      } else if (data.mode === "outfit") {
        setStatus(t.outfit);
        renderGrid(data.results, "outfit");
      } else {
        setStatus(t.similar);
        renderGrid(data.results, "shoe");
      }
    }

    function renderGrid(items, mode) {
      resultsEl.innerHTML = "";
      if (!items || items.length === 0) {
        const p = document.createElement("p");
        p.className = "khriss__empty";
        p.textContent = t.empty;
        resultsEl.appendChild(p);
        return;
      }
      items.forEach((p) => resultsEl.appendChild(card(p, mode)));
    }

    /* Placeholder cards while the request is in flight: the wait is a couple of
       seconds of CLIP work, and an empty box reads as broken. */
    function showSkeletons() {
      resultsEl.innerHTML = "";
      for (let i = 0; i < 4; i++) {
        const el = document.createElement("div");
        el.className = "khriss__card khriss__card--skeleton";
        el.innerHTML =
          '<div class="khriss__sk khriss__sk-img"></div>' +
          '<div class="khriss__sk khriss__sk-line"></div>' +
          '<div class="khriss__sk khriss__sk-line khriss__sk-line--short"></div>';
        resultsEl.appendChild(el);
      }
    }

    function card(p, mode) {
      const el = document.createElement("div");
      el.className = "khriss__card";
      el.setAttribute("role", "listitem");

      el.innerHTML = `
        <a class="khriss__card-link" href="/products/${encodeURIComponent(p.handle || "")}">
          <div class="khriss__card-media">
            <img class="khriss__card-img" src="${escapeHtml(p.image_url || "")}" alt="${escapeHtml(p.title)}" loading="lazy" />
          </div>
          <div class="khriss__card-title">${escapeHtml(p.title)}</div>
          <div class="khriss__card-price">${escapeHtml(formatMoney(p.price, money))}</div>
        </a>
        ${mode === "outfit" && p.rationale ? `<p class="khriss__rationale">${escapeHtml(p.rationale)}</p>` : ""}
        <button type="button" class="khriss__add" ${p.variant_id ? "" : "disabled"}>${escapeHtml(t.add)}</button>
      `;

      const addBtn = el.querySelector(".khriss__add");
      addBtn.addEventListener("click", () => addToCart(p, addBtn, t));
      return el;
    }

    function setTabsVisible(on) {
      tabs.hidden = !on;
    }

    function setStatus(msg, opts) {
      status.hidden = false;
      status.textContent = msg;
      status.classList.toggle("is-busy", !!(opts && opts.busy));
    }
  }

  function readStrings(root) {
    const d = root.dataset;
    return {
      analysing: d.khrissTAnalysing,
      outfit: d.khrissTOutfit,
      similar: d.khrissTSimilar,
      both: d.khrissTBoth,
      empty: d.khrissTEmpty,
      error: d.khrissTError,
      add: d.khrissTAdd,
      adding: d.khrissTAdding,
      added: d.khrissTAdded,
      retry: d.khrissTRetry,
    };
  }

  /* Render through the shop's own money_format so the widget matches the rest
     of the storefront (dirhams here, not a hardcoded dollar sign). The format
     string carries an {{amount}}-style placeholder; everything else is literal. */
  function formatMoney(value, format) {
    if (value === null || value === undefined || value === "") return "";
    const n = Number(value);
    if (!isFinite(n)) return String(value);
    const amount = n.toLocaleString("fr-FR", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    const out = format.replace(/\{\{\s*amount[a-z_]*\s*\}\}/gi, amount);
    // No placeholder in the format (or none configured) -> append the number.
    return out === format ? `${amount} ${format}`.trim() : out;
  }

  async function postImage(api, file) {
    const form = new FormData();
    form.append("image", file);
    const res = await fetch(`${api}/recommend`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async function addToCart(product, btn, t) {
    if (!product.variant_id) return;
    btn.disabled = true;
    btn.textContent = t.adding;
    try {
      const res = await fetch("/cart/add.js", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: product.variant_id, quantity: 1 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      btn.textContent = t.added;
      btn.classList.add("is-added");
    } catch (e) {
      btn.textContent = t.retry;
      btn.disabled = false;
      console.error("khriss add-to-cart:", e);
    }
  }

  function escapeHtml(s) {
    return String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }
})();
