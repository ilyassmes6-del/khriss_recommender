/* khriss recommender — client widget.
 *
 * Flow: user picks an image -> show preview -> POST to {api}/recommend ->
 * render a results grid. Mode A ("outfit") shows a stylist rationale under each
 * card; Mode B ("shoe") shows a plain similarity grid; ambiguous ("both")
 * renders two tabs backed by the two result lists in the response.
 */
(function () {
  const roots = document.querySelectorAll(".khriss");
  roots.forEach(initWidget);

  function initWidget(root) {
    const api = (root.dataset.khrissApi || "").replace(/\/$/, "");
    const fileInput = root.querySelector(".khriss__file");
    const preview = root.querySelector(".khriss__preview");
    const previewImg = root.querySelector(".khriss__preview-img");
    const status = root.querySelector(".khriss__status");
    const tabs = root.querySelector(".khriss__tabs");
    const resultsEl = root.querySelector(".khriss__results");

    let lastResponse = null;

    fileInput.addEventListener("change", async () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;

      // Client-side preview.
      previewImg.src = URL.createObjectURL(file);
      preview.hidden = false;
      resultsEl.innerHTML = "";
      tabs.hidden = true;
      setStatus("Analysing your photo…");

      try {
        const data = await postImage(api, file);
        lastResponse = data;
        render(data);
      } catch (err) {
        setStatus("Something went wrong. Please try another photo.");
        console.error("khriss:", err);
      }
    });

    // Tab switching for ambiguous results.
    tabs.querySelectorAll(".khriss__tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        tabs.querySelectorAll(".khriss__tab").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        const which = btn.dataset.tab;
        renderGrid(which === "outfit" ? lastResponse.outfit_results : lastResponse.shoe_results, which);
      });
    });

    function render(data) {
      if (data.mode === "both") {
        setStatus("Not sure if that's a shoe or an outfit — here are both.");
        tabs.hidden = false;
        renderGrid(data.outfit_results || [], "outfit");
      } else if (data.mode === "outfit") {
        setStatus("Shoes to complete your look:");
        renderGrid(data.results, "outfit");
      } else {
        setStatus("Visually similar shoes:");
        renderGrid(data.results, "shoe");
      }
    }

    function renderGrid(items, mode) {
      resultsEl.innerHTML = "";
      if (!items || items.length === 0) {
        resultsEl.innerHTML = '<p class="khriss__empty">No matches found.</p>';
        return;
      }
      items.forEach((p) => resultsEl.appendChild(card(p, mode)));
    }

    function card(p, mode) {
      const el = document.createElement("div");
      el.className = "khriss__card";
      el.setAttribute("role", "listitem");

      const price = p.price ? `$${p.price}` : "";
      el.innerHTML = `
        <a class="khriss__card-link" href="/products/${p.handle}">
          <img class="khriss__card-img" src="${p.image_url || ""}" alt="${escapeHtml(p.title)}" loading="lazy" />
          <div class="khriss__card-title">${escapeHtml(p.title)}</div>
          <div class="khriss__card-price">${price}</div>
        </a>
        ${mode === "outfit" && p.rationale ? `<p class="khriss__rationale">${escapeHtml(p.rationale)}</p>` : ""}
        <button class="khriss__add" ${p.variant_id ? "" : "disabled"}>Add to cart</button>
      `;

      const addBtn = el.querySelector(".khriss__add");
      addBtn.addEventListener("click", () => addToCart(p, addBtn));
      return el;
    }

    function setStatus(msg) {
      status.hidden = false;
      status.textContent = msg;
    }
  }

  async function postImage(api, file) {
    const form = new FormData();
    form.append("image", file);
    const res = await fetch(`${api}/recommend`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async function addToCart(product, btn) {
    if (!product.variant_id) return;
    btn.disabled = true;
    btn.textContent = "Adding…";
    try {
      const res = await fetch("/cart/add.js", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: product.variant_id, quantity: 1 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      btn.textContent = "Added ✓";
    } catch (e) {
      btn.textContent = "Try again";
      btn.disabled = false;
      console.error("khriss add-to-cart:", e);
    }
  }

  function escapeHtml(s) {
    return String(s || "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }
})();
