/* khriss / Masilya Match — client widget.
 *
 * Flow: the shopper picks an image in the hero ("Envoie une photo. Complète ton
 * look.") -> preview + scan there -> POST to {api}/recommend -> the dark
 * "IT'S A MATCH" panel below unhides and fills with the suggestions.
 *
 * Mode A ("outfit") groups the results into one row per category (Chaussures /
 * Sacs / Bijoux), each with several articles and a stylist rationale; Mode B
 * ("shoe") shows a flat similarity grid; ambiguous ("both") renders two tabs
 * backed by the two result lists in the response.
 *
 * All copy comes from data-khriss-* attributes set in the Liquid block, so this
 * file carries no user-facing strings and stays translatable.
 */
(function () {
  // Category display order + labels come from the block; drives the row order.
  const CATEGORY_ORDER = ["shoes", "bags", "jewelry"];
  const SIZE_KEY = "khriss:size";

  const roots = document.querySelectorAll(".khriss");
  roots.forEach(initWidget);

  function initWidget(root) {
    const api = (root.dataset.khrissApi || "").replace(/\/$/, "");
    const t = readStrings(root);
    const catLabels = {
      shoes: root.dataset.khrissCatShoes || "",
      bags: root.dataset.khrissCatBags || "",
      jewelry: root.dataset.khrissCatJewelry || "",
    };
    const money = root.dataset.khrissMoney || "{{amount}}";

    const fileInput = root.querySelector(".khriss__file");
    const drop = root.querySelector(".khriss__drop");
    const sizeButtons = root.querySelectorAll(".khriss__size");
    const preview = root.querySelector(".khriss__preview");
    const previewImg = root.querySelector(".khriss__preview-img");
    const status = root.querySelector(".khriss__status");
    const demo = root.querySelector(".khriss__demo");
    const resultHead = root.querySelector(".khriss__result-head");
    const tabs = root.querySelector(".khriss__tabs");
    const resultsEl = root.querySelector(".khriss__results");

    let lastResponse = null;
    let pending = false;
    let size = readStoredSize();

    /* --- size ---------------------------------------------------------- */
    // Remembered across visits: a shopper's shoe size is the one thing about
    // them that does not change between sessions, and asking again every time
    // is the kind of friction that gets the picker ignored.
    function readStoredSize() {
      try {
        return window.localStorage.getItem(SIZE_KEY) || "";
      } catch (e) {
        return ""; // private mode / storage disabled
      }
    }

    function storeSize(value) {
      try {
        if (value) window.localStorage.setItem(SIZE_KEY, value);
        else window.localStorage.removeItem(SIZE_KEY);
      } catch (e) {
        /* not worth breaking the widget over */
      }
    }

    function setSize(value) {
      size = value || "";
      sizeButtons.forEach((b) => {
        const on = (b.dataset.size || "") === size;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
      storeSize(size);
    }

    sizeButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        setSize(btn.dataset.size || "");
        // Re-run against the new size rather than leaving stale results that
        // no longer match the chip the shopper just pressed.
        if (lastFile && !pending) handleFile(lastFile);
      });
    });
    // Restore the remembered choice, but only if that size is still offered.
    if (size && ![...sizeButtons].some((b) => (b.dataset.size || "") === size)) {
      size = "";
    }
    setSize(size);

    let lastFile = null;

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
      lastFile = file; // so a size change can re-run the same photo

      previewImg.src = URL.createObjectURL(file);
      preview.hidden = false;
      preview.classList.add("is-analysing");
      // The drop zone stays visible so the shopper can try another photo (and
      // so an error still leaves a working control) -- the preview sits below it.
      lastResponse = null;
      setTabsVisible(false);
      setStatus(t.analysing, { busy: true });

      // The panel fills in below while the shopper stays in the hero watching
      // the scan; we deliberately do NOT scroll here. Jumping to an empty
      // "IT'S A MATCH" before anything has been found reads as broken -- the
      // scroll happens in render(), once there are actual suggestions to see.
      demo.hidden = false;
      setHead(t.analysing);
      showSkeletons();

      try {
        const data = await postImage(api, file, size);
        lastResponse = data;
        render(data);
      } catch (err) {
        resultsEl.innerHTML = "";
        setHead(t.error);
        setStatus(t.error);
        console.error("khriss:", err);
      } finally {
        preview.classList.remove("is-analysing");
        pending = false;
      }
    }

    /* Bring the dark panel into view: the upload sits up in the hero, so on a
       long page the results would otherwise land well below the fold. */
    function scrollToResults() {
      const reduced =
        window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      try {
        demo.scrollIntoView({
          behavior: reduced ? "auto" : "smooth",
          block: "start",
        });
      } catch (e) {
        demo.scrollIntoView(); // older browsers: no options object
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
        if (which === "outfit") {
          renderGrouped(lastResponse.outfit_results || []); // owns the head
        } else {
          setHead(t.similar);
          renderFlat(lastResponse.shoe_results || [], "shoe");
        }
      });
    });

    function render(data) {
      let found;
      if (data.mode === "both") {
        // Tabs let the shopper switch between the two readings; the grouped
        // "complete the look" view is shown first. renderGrouped owns the head
        // (the suggestion count), so nothing sets it here.
        setTabsVisible(true);
        found = renderGrouped(data.outfit_results || []);
      } else if (data.mode === "outfit") {
        found = renderGrouped(data.results || []);
      } else {
        setHead(t.similar);
        found = renderFlat(data.results || [], "shoe");
      }

      if (found) {
        // Only now is there something worth jumping to.
        setStatus("");
        status.hidden = true;
        scrollToResults();
      } else {
        // Nothing found: say so where the shopper already is, next to the photo
        // they just sent, rather than sending them to an empty panel.
        setStatus(size ? t.emptySize : t.empty);
      }
    }

    /* Outfit path: one row per category, several articles each.
       Returns whether anything was rendered, so the caller knows if there is
       something worth scrolling to. */
    function renderGrouped(items) {
      resultsEl.innerHTML = "";
      if (!items.length) {
        showEmpty();
        setHead("");
        return false;
      }
      setHead(t.found.replace("%n%", items.length));

      const byCat = new Map();
      items.forEach((p) => {
        const key = p.category || "_other";
        if (!byCat.has(key)) byCat.set(key, []);
        byCat.get(key).push(p);
      });

      // Known categories first in a fixed order, then anything unmapped.
      const keys = CATEGORY_ORDER.filter((k) => byCat.has(k));
      byCat.forEach((_, k) => {
        if (!keys.includes(k)) keys.push(k);
      });

      keys.forEach((key) => {
        const group = byCat.get(key);
        resultsEl.appendChild(
          categoryGroup(catLabels[key] || "", group, "outfit")
        );
      });
      return true;
    }

    /* Similar-items path: a single flat grid, no category headings. */
    function renderFlat(items, mode) {
      resultsEl.innerHTML = "";
      if (!items.length) {
        showEmpty();
        return false;
      }
      const grid = document.createElement("div");
      grid.className = "khriss__grid";
      grid.setAttribute("role", "list");
      items.forEach((p) => grid.appendChild(card(p, mode)));
      resultsEl.appendChild(grid);
      return true;
    }

    function categoryGroup(label, items, mode) {
      const wrap = document.createElement("div");
      wrap.className = "khriss__group";
      if (label) {
        const head = document.createElement("div");
        head.className = "khriss__group-head";
        head.innerHTML =
          escapeHtml(label) +
          `<span class="khriss__group-count">${items.length}</span>`;
        wrap.appendChild(head);
      }
      const grid = document.createElement("div");
      grid.className = "khriss__grid";
      grid.setAttribute("role", "list");
      items.forEach((p) => grid.appendChild(card(p, mode)));
      wrap.appendChild(grid);
      return wrap;
    }

    function showEmpty() {
      const p = document.createElement("p");
      p.className = "khriss__empty";
      // With a size set, "no results" usually means "none in this size" -- say
      // so, or the shopper retries photos when the fix is another size.
      p.textContent = size ? t.emptySize : t.empty;
      resultsEl.appendChild(p);
    }

    /* Placeholder cards while the request is in flight: the wait is a couple of
       seconds of CLIP work, and an empty box reads as broken. */
    function showSkeletons() {
      resultsEl.innerHTML = "";
      const grid = document.createElement("div");
      grid.className = "khriss__grid";
      for (let i = 0; i < 6; i++) {
        const el = document.createElement("div");
        el.className = "khriss__card khriss__card--skeleton";
        el.innerHTML =
          '<div class="khriss__sk khriss__sk-img"></div>' +
          '<div class="khriss__sk khriss__sk-line"></div>' +
          '<div class="khriss__sk khriss__sk-line khriss__sk-line--short"></div>';
        grid.appendChild(el);
      }
      resultsEl.appendChild(grid);
    }

    function card(p, mode) {
      const el = document.createElement("div");
      el.className = "khriss__card";
      el.setAttribute("role", "listitem");

      const cat = p.category && catLabels[p.category] ? catLabels[p.category] : "";

      el.innerHTML = `
        <span class="khriss__tag">✦ ${escapeHtml(t.match)}</span>
        <a class="khriss__card-link" href="/products/${encodeURIComponent(p.handle || "")}">
          <div class="khriss__card-media">
            <img class="khriss__card-img" src="${escapeHtml(p.image_url || "")}" alt="${escapeHtml(p.title)}" loading="lazy" />
          </div>
        </a>
        <div class="khriss__card-body">
          ${cat ? `<div class="khriss__card-cat">${escapeHtml(cat)}</div>` : ""}
          <a class="khriss__card-link" href="/products/${encodeURIComponent(p.handle || "")}">
            <div class="khriss__card-title">${escapeHtml(p.title)}</div>
          </a>
          <div class="khriss__card-price">${escapeHtml(formatMoney(p.price, money))}</div>
          ${p.size ? `<span class="khriss__card-size">${escapeHtml(t.inSize.replace("%s%", p.size))}</span>` : ""}
          ${mode === "outfit" && p.rationale ? `<p class="khriss__rationale">${escapeHtml(p.rationale)}</p>` : ""}
          <button type="button" class="khriss__add" ${p.variant_id ? "" : "disabled"}>${escapeHtml(t.add)}</button>
        </div>
      `;

      const addBtn = el.querySelector(".khriss__add");
      addBtn.addEventListener("click", () => addToCart(p, addBtn, t));
      return el;
    }

    function setTabsVisible(on) {
      tabs.hidden = !on;
    }

    function setHead(msg) {
      if (!msg) {
        resultHead.hidden = true;
        resultHead.textContent = "";
        return;
      }
      resultHead.hidden = false;
      resultHead.textContent = "✦ " + msg;
    }

    function setStatus(msg, opts) {
      if (!msg) {
        status.textContent = "";
        status.classList.remove("is-busy");
        return;
      }
      status.hidden = false;
      status.textContent = msg;
      status.classList.toggle("is-busy", !!(opts && opts.busy));
    }
  }

  function readStrings(root) {
    const d = root.dataset;
    return {
      analysing: d.khrissTAnalysing,
      found: d.khrissTFound || "%n%", // %n% is replaced with the result count
      similar: d.khrissTSimilar,
      empty: d.khrissTEmpty,
      emptySize: d.khrissTEmptySize || d.khrissTEmpty,
      inSize: d.khrissTInSize || "%s%",
      error: d.khrissTError,
      match: d.khrissTMatch,
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

  async function postImage(api, file, size) {
    const form = new FormData();
    form.append("image", file);
    // Omitted entirely when unset, so the API keeps its no-filter default.
    if (size) form.append("size", size);
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
