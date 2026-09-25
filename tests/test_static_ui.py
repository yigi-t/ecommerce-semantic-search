from pathlib import Path
import re
import unittest

from lxml import html

from src.outfit import (
    BOTTOM_CATEGORIES,
    LAYER_CATEGORIES,
    ONE_PIECE_CATEGORIES,
    TOP_CATEGORIES,
)


UI_PATH = Path(__file__).resolve().parents[1] / "static" / "index.html"


class StaticUiContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = UI_PATH.read_text(encoding="utf-8")
        cls.document = html.fromstring(cls.source)

    def test_existing_search_surface_and_request_are_preserved(self):
        self.assertTrue(self.document.xpath('//form[@id="f"]'))
        self.assertTrue(self.document.xpath('//*[@id="q"]'))
        self.assertTrue(self.document.xpath('//*[@id="meta"]'))
        self.assertTrue(self.document.xpath('//*[@id="grid"]'))
        self.assertIn("`/search?q=${encodeURIComponent(query)}&limit=24`", self.source)

    def test_outfit_panel_starts_hidden_and_is_loaded_lazily(self):
        self.assertTrue(self.document.xpath('//main//section[@id="outfit" and @hidden]'))
        self.assertEqual(self.source.count("outfit-recommendations?limit=4"), 1)
        self.assertIn("Bununla kombinle", self.source)

    def test_selected_product_visual_is_rendered_in_outfit_panel(self):
        self.assertIn("source-media", self.source)
        self.assertIn("product?.image", self.source)
        self.assertIn("Seçtiğin parça", self.source)
        self.assertIn("setSourceProduct(data.source)", self.source)

    def test_outfit_panel_is_anchored_without_adding_a_grid_row(self):
        self.assertIn("#grid > .outfit", self.source)
        self.assertIn("position:absolute", self.source)
        self.assertIn("function positionOutfitAtCard(button)", self.source)
        self.assertIn("selectedCard.after(outfitSection)", self.source)
        self.assertIn("showOutfitPanel(sourceProduct, button)", self.source)
        self.assertIn("positionOutfitAtCard(activeOutfitButton)", self.source)
        self.assertIn("meta.after(outfitSection)", self.source)
        self.assertNotIn("outfitSection.scrollIntoView", self.source)
        outfit_rule = re.search(
            r"#grid > \.outfit\s*\{(.*?)\}", self.source, re.DOTALL
        )
        self.assertIsNotNone(outfit_rule)
        self.assertNotIn("grid-column:1 / -1", outfit_rule.group(1))

    def test_anchored_outfit_panel_has_dialog_accessibility(self):
        self.assertTrue(self.document.xpath(
            '//section[@id="outfit" and @role="dialog" and @aria-modal="false"]'
        ))
        self.assertIn("button.setAttribute('aria-haspopup', 'dialog')", self.source)
        self.assertIn("event.key !== 'Escape'", self.source)

    def test_clicking_outside_closes_the_outfit_panel(self):
        self.assertIn("function closeOutfitOnOutsideClick(event)", self.source)
        self.assertIn("outfitSection.contains(event.target)", self.source)
        self.assertIn("event.target.closest?.('button[data-outfit-id]')", self.source)
        self.assertIn(
            "document.addEventListener('click', closeOutfitOnOutsideClick)",
            self.source,
        )

    def test_size_preferences_are_inline_accessible_and_local(self):
        self.assertTrue(self.document.xpath(
            '//*[@id="size-preference-toggle" and @type="button" '
            'and @aria-controls="size-preference-panel" and @aria-expanded="false"]'
        ))
        self.assertTrue(self.document.xpath(
            '//*[@id="size-preference-panel" and @hidden]'
        ))
        self.assertEqual(len(self.document.xpath(
            '//select[@data-size-preference]'
        )), 6)
        self.assertIn(
            "const SIZE_PREFERENCES_STORAGE_KEY = 'defacto-size-preferences-v1'",
            self.source,
        )
        self.assertIn("localStorage.getItem(SIZE_PREFERENCES_STORAGE_KEY)", self.source)
        self.assertIn("localStorage.setItem(SIZE_PREFERENCES_STORAGE_KEY", self.source)
        self.assertIn("localStorage.removeItem(SIZE_PREFERENCES_STORAGE_KEY)", self.source)
        self.assertIn("function sanitizeSizePreferences(value)", self.source)
        self.assertIn("sizePreferences = loadSizePreferences()", self.source)

    def test_remembered_sizes_update_cards_without_hiding_search_results(self):
        self.assertIn("function rememberedSizeStatus(product)", self.source)
        self.assertIn(
            "appendRememberedSizeStatus(body, product, { historical:historicalSizeStatus })",
            self.source,
        )
        self.assertIn("appendRememberedSizeStatus(copy, product)", self.source)
        self.assertIn("Beden bilgisi yok", self.source)
        self.assertIn("Standart beden", self.source)
        self.assertIn("Bedenin ${sizePreferences.alpha} stokta", self.source)
        self.assertIn("function refreshProductsForSizePreference()", self.source)
        self.assertIn("renderActiveProductView()", self.source)
        self.assertIn("historicalSizeStatus:true", self.source)
        self.assertIn("Son görüntülendiğinde:", self.source)
        self.assertNotIn("searchProducts = searchProducts.filter", self.source)
        self.assertIn("`/search?q=${encodeURIComponent(query)}&limit=24`", self.source)

    def test_remembered_sizes_are_sent_to_initial_and_edited_outfits(self):
        self.assertIn("function preferredOutfitSizesFor(source)", self.source)
        self.assertIn("function appendPreferredOutfitSizes(params, source)", self.source)
        self.assertIn("appendPreferredOutfitSizes(params, source)", self.source)
        self.assertIn(
            "appendPreferredOutfitSizes(preferredSizeParams, sourceProduct)",
            self.source,
        )
        self.assertIn("params.append('preferred_size', size)", self.source)
        self.assertIn("outfitCacheKey(productId, sourceProduct)", self.source)

    def test_size_systems_are_kept_separate_and_combined_sizes_are_supported(self):
        self.assertIn("'alpha','numeric','shoe','jeanWaist','jeanLength','child'", self.source)
        self.assertIn("FOOTWEAR_CATEGORIES.has(product?.category)", self.source)
        self.assertIn("function parseJeanSize(value)", self.source)
        self.assertIn("function normalizeChildSize(value)", self.source)
        self.assertIn("function childSizeMatches(value, preference)", self.source)
        self.assertIn("function alphaSizeAliases(preference)", self.source)
        self.assertIn("M:['M','S/M','M/L']", self.source)
        self.assertIn("'Sneaker','Bot','Çizme','Babet','Loafer','Çorap'", self.source)
        self.assertIn("SIZE_FILTER_EXEMPT_CATEGORIES.has(product?.category)", self.source)
        self.assertIn("'Kemer','Çanta','Takı','Yüzük'", self.source)

    def test_footwear_ranges_and_historical_snapshots_are_accurate(self):
        self.assertIn("{ slashIsRange = false }", self.source)
        self.assertIn("{ slashIsRange:true }", self.source)
        self.assertIn("' stokta değildi'", self.source)
        self.assertIn("product.sizes.slice()", self.source)
        self.assertNotIn("product.sizes.slice(0, 20)", self.source)

    def test_incomplete_jean_size_is_not_treated_as_a_saved_preference(self):
        self.assertIn(
            "(preferences?.jeanWaist && preferences?.jeanLength)",
            self.source,
        )
        self.assertIn("if (!storedPreferences.jeanWaist || !storedPreferences.jeanLength)", self.source)
        self.assertIn("storedPreferences.jeanWaist = ''", self.source)
        self.assertIn("storedPreferences.jeanLength = ''", self.source)
        self.assertIn("Jean bedeni için bel ve boyu birlikte seçmelisin.", self.source)

    def test_size_preferences_handle_cross_tab_clear_and_touch_targets(self):
        self.assertIn(
            "event.key !== SIZE_PREFERENCES_STORAGE_KEY && event.key !== null",
            self.source,
        )
        self.assertIn(".size-field select {", self.source)
        self.assertIn("min-height:44px", self.source)

    def test_recent_products_tab_is_local_persistent_and_bounded(self):
        self.assertTrue(self.document.xpath('//*[@id="product-tabs" and @role="tablist"]'))
        self.assertTrue(self.document.xpath('//*[@id="recent-products-tab" and @role="tab"]'))
        self.assertIn("const RECENT_STORAGE_KEY = 'defacto-recent-products-v1'", self.source)
        self.assertIn("const RECENT_LIMIT = 10", self.source)
        self.assertIn("localStorage.getItem(RECENT_STORAGE_KEY)", self.source)
        self.assertIn("localStorage.setItem(RECENT_STORAGE_KEY", self.source)
        self.assertIn(".slice(0, RECENT_LIMIT)", self.source)
        self.assertIn("saveRecentProducts();", self.source)
        self.assertIn("updateRecentProductsTab();", self.source)

    def test_recent_products_can_be_cleared_with_a_button(self):
        self.assertTrue(self.document.xpath(
            '//*[@id="clear-recent-products" and @type="button" and @hidden]'
        ))
        self.assertIn("function clearRecentProducts()", self.source)
        self.assertIn("localStorage.removeItem(RECENT_STORAGE_KEY)", self.source)
        self.assertIn(
            "clearRecentProductsButton.addEventListener('click', clearRecentProducts)",
            self.source,
        )
        self.assertIn("clearRecentProductsButton.disabled = count === 0", self.source)

    def test_cleared_recent_products_can_be_undone(self):
        self.assertTrue(self.document.xpath(
            '//*[@id="recent-clear-undo" and @role="group" and @hidden]'
        ))
        self.assertTrue(self.document.xpath(
            '//*[@id="undo-clear-recent" and @type="button"]'
        ))
        self.assertIn("const RECENT_UNDO_TIMEOUT_MS = 10000", self.source)
        self.assertIn("const RECENT_UNDO_ANIMATION_MS = 320", self.source)
        self.assertIn("const clearedProducts = recentProducts.slice()", self.source)
        self.assertIn("showRecentClearUndo(clearedProducts)", self.source)
        self.assertIn("function undoClearRecentProducts()", self.source)
        self.assertIn(
            "recentProducts = [...recentProducts, ...restoredProducts].slice(0, RECENT_LIMIT)",
            self.source,
        )
        self.assertIn("currentIds.add(productId)", self.source)
        self.assertIn(
            "undoClearRecentButton.addEventListener('click', undoLastClear)",
            self.source,
        )
        self.assertIn("recentClearUndo.classList.add('is-visible')", self.source)
        self.assertIn("recentClearUndo.classList.add('is-hiding')", self.source)
        self.assertIn("window.setTimeout(finishHiding, RECENT_UNDO_ANIMATION_MS)", self.source)

    def test_saved_outfits_reuse_the_animated_clear_undo(self):
        self.assertEqual(self.source.count('id="undo-clear-recent"'), 1)
        self.assertIn("function showClearUndo(type, message, announcement)", self.source)
        self.assertIn("const clearedOutfits = savedOutfits.slice()", self.source)
        self.assertIn("showSavedOutfitsClearUndo(clearedOutfits)", self.source)
        self.assertIn("function undoClearSavedOutfits()", self.source)
        self.assertIn(
            "savedOutfits = [...savedOutfits, ...restoredOutfits].slice(0, SAVED_OUTFITS_LIMIT)",
            self.source,
        )
        self.assertIn("if (clearUndoType === 'recent')", self.source)
        self.assertIn("if (clearUndoType === 'saved')", self.source)
        self.assertIn("'Kaydedilen kombinler temizlendi.'", self.source)
        self.assertIn("'Kaydedilen kombinler geri getirildi.'", self.source)

    def test_removing_one_saved_outfit_reuses_the_same_undo(self):
        self.assertEqual(self.source.count('id="undo-clear-recent"'), 1)
        self.assertIn("function showRemovedSavedOutfitUndo(outfit, index)", self.source)
        self.assertIn("showRemovedSavedOutfitUndo(removedOutfit, removedIndex)", self.source)
        self.assertIn("showRemovedSavedOutfitUndo(removedOutfit, savedIndex)", self.source)
        self.assertIn("function undoRemovedSavedOutfit()", self.source)
        self.assertIn("restoredOutfits.splice(insertIndex, 0, removed.outfit)", self.source)
        self.assertIn("if (clearUndoType === 'saved-item')", self.source)
        self.assertIn("'Kombin kayıttan kaldırıldı. 10 saniye içinde geri alabilirsiniz.'", self.source)
        self.assertIn("'Kombin yeniden kaydedildi.'", self.source)

    def test_product_visits_are_deduplicated_and_keep_existing_cards(self):
        self.assertIn("link.addEventListener('click', () => rememberProduct(product))", self.source)
        self.assertIn("recentProducts.filter(item => String(item.id) !== snapshot.id)", self.source)
        self.assertIn(
            "renderProductGrid(recentProducts, { historicalSizeStatus:true })",
            self.source,
        )
        self.assertIn("setProductView('search', { render:false })", self.source)
        self.assertIn("`/search?q=${encodeURIComponent(query)}&limit=24`", self.source)

    def test_outfit_actions_also_record_the_source_as_recent(self):
        self.assertIn(
            "rememberProduct(sourceProduct, { refreshRecent:false })",
            self.source,
        )
        self.assertIn(
            "function rememberProduct(product, { refreshRecent = true } = {})",
            self.source,
        )

    def test_saved_outfits_are_local_persistent_bounded_and_deduplicated(self):
        self.assertTrue(self.document.xpath(
            '//*[@id="saved-outfits-tab" and @role="tab" and @data-view="saved"]'
        ))
        self.assertIn(
            "const SAVED_OUTFITS_STORAGE_KEY = 'defacto-saved-outfits-v1'",
            self.source,
        )
        self.assertIn("const SAVED_OUTFITS_LIMIT = 20", self.source)
        self.assertIn(
            "localStorage.getItem(SAVED_OUTFITS_STORAGE_KEY)",
            self.source,
        )
        self.assertIn(
            "localStorage.setItem(SAVED_OUTFITS_STORAGE_KEY",
            self.source,
        )
        self.assertIn(
            "JSON.stringify([sourceId, productId].sort())",
            self.source,
        )
        self.assertIn(".slice(0, SAVED_OUTFITS_LIMIT)", self.source)

    def test_outfit_recommendations_can_be_saved_as_product_pairs(self):
        self.assertIn(
            "createOutfitRecommendationCard(data.source, product, index)",
            self.source,
        )
        self.assertIn("function toggleSavedOutfit(source, product)", self.source)
        self.assertIn("button.setAttribute('aria-pressed', String(isSaved))", self.source)
        self.assertIn("isSaved ? 'Kaydı kaldır' : 'Kombini kaydet'", self.source)
        self.assertIn("button.addEventListener('click', () => toggleSavedOutfit", self.source)
        self.assertIn("source:sourceSnapshot", self.source)
        self.assertIn("product:productSnapshot", self.source)
        self.assertIn("reason:String(product?.reason", self.source)

    def test_outfit_recommendations_are_editable_per_card(self):
        self.assertIn("const OUTFIT_EDIT_OPTIONS = {", self.source)
        self.assertIn("alternative:{", self.source)
        self.assertIn("cheaper:{", self.source)
        self.assertIn("different_color:{", self.source)
        self.assertIn("editActions.setAttribute('role', 'group')", self.source)
        self.assertIn("editButton.dataset.outfitPreference = preference", self.source)
        self.assertIn("editButton.dataset.outfitEditIndex = String(index)", self.source)
        self.assertIn("editButton.dataset.currentProductId = String(product.id)", self.source)
        self.assertIn("'Öneriyi değiştir'", self.source)
        self.assertIn("'Daha uygun fiyatlı'", self.source)
        self.assertIn("'Başka renk'", self.source)

    def test_outfit_edit_replaces_only_the_selected_card(self):
        self.assertIn("async function replaceOutfitRecommendation(button)", self.source)
        self.assertIn("params.set('limit', '1')", self.source)
        self.assertIn("params.set('preference', preference)", self.source)
        self.assertIn("params.set('current_product_id', String(current.id))", self.source)
        self.assertIn("params.append('exclude_product_id', productId)", self.source)
        self.assertIn("activeOutfitData.products[index] = replacement", self.source)
        self.assertIn(
            "createOutfitRecommendationCard(source, replacement, index)",
            self.source,
        )
        self.assertIn("card.replaceWith(replacementCard)", self.source)
        self.assertIn("setOutfitEditBusy(replacementCard, false)", self.source)
        self.assertIn("Mevcut öneri korunuyor.", self.source)
        self.assertIn(
            "outfitGrid.addEventListener('click', event => {",
            self.source,
        )

        edit_function = re.search(
            r"async function replaceOutfitRecommendation\(button\)(.*?)\n}\n\nasync function loadOutfit",
            self.source,
            re.DOTALL,
        )
        self.assertIsNotNone(edit_function)
        self.assertNotIn("outfitCache.set", edit_function.group(1))

    def test_outfit_edit_remembers_each_slots_products_without_repeating_them(self):
        self.assertIn("const OUTFIT_EDIT_EXCLUDE_LIMIT = 100", self.source)
        self.assertIn("let outfitProductHistoryByIndex = new Map()", self.source)
        self.assertNotIn("outfitCheaperReferenceIds", self.source)
        self.assertIn(
            "new Set([...productHistory, ...visibleProductIds])",
            self.source,
        )
        edit_function = re.search(
            r"async function replaceOutfitRecommendation\(button\)(.*?)\n}\n\nasync function loadOutfit",
            self.source,
            re.DOTALL,
        )
        self.assertIsNotNone(edit_function)
        self.assertNotIn(".slice(0, 20)", edit_function.group(1))
        self.assertIn("productHistory.add(replacementId)", self.source)
        self.assertIn(
            "productHistory.has(replacementId) || visibleProductIds.includes(replacementId)",
            self.source,
        )

    def test_cheaper_edit_always_compares_with_the_current_visible_product(self):
        self.assertIn(
            "params.set('current_product_id', String(current.id))",
            self.source,
        )
        self.assertIn(
            "empty:'Daha uygun fiyatlı başka bir uyumlu parça bulunamadı.'",
            self.source,
        )

    def test_outfit_edit_controls_remain_usable_on_small_screens(self):
        mobile_rule = re.search(
            r"@media \(max-width:520px\)\s*\{(.*?)\n  \}",
            self.source,
            re.DOTALL,
        )
        self.assertIsNotNone(mobile_rule)
        self.assertIn(".outfit-edit-actions { grid-template-columns:1fr", mobile_rule.group(1))
        self.assertIn(".outfit-edit-button { min-height:44px; }", mobile_rule.group(1))

    def test_outfit_edit_requests_are_cancelled_when_panel_closes(self):
        hide_function = re.search(
            r"function hideOutfit\(restoreFocus = false\)(.*?)\n}\n\nfunction showOutfitPanel",
            self.source,
            re.DOTALL,
        )
        self.assertIsNotNone(hide_function)
        self.assertIn("outfitEditController.abort()", hide_function.group(1))
        self.assertIn("activeOutfitData = null", hide_function.group(1))
        self.assertIn("outfitProductHistoryByIndex = new Map()", hide_function.group(1))

    def test_outfit_edit_controls_are_reenabled_for_repeated_clicks(self):
        edit_function = re.search(
            r"async function replaceOutfitRecommendation\(button\)(.*?)\n}\n\nasync function loadOutfit",
            self.source,
            re.DOTALL,
        )
        self.assertIsNotNone(edit_function)
        self.assertIn("function currentOutfitEditButton(index, preference)", self.source)
        self.assertIn("if (outfitEditController === controller)", edit_function.group(1))
        self.assertIn("setOutfitEditBusy(null, false)", edit_function.group(1))
        self.assertIn(
            "currentOutfitEditButton(index, preference)?.focus",
            edit_function.group(1),
        )

    def test_personalized_tab_is_accessible_and_in_the_requested_order(self):
        tab_ids = self.document.xpath(
            '//*[@id="product-tabs"]/*[@role="tab"]/@id'
        )
        self.assertEqual(tab_ids, [
            "search-results-tab",
            "personalized-products-tab",
            "recent-products-tab",
            "saved-outfits-tab",
        ])
        self.assertTrue(self.document.xpath(
            '//*[@id="personalized-products-tab" and @type="button" '
            'and @data-view="personalized" and @aria-controls="grid"]'
        ))
        self.assertTrue(self.document.xpath(
            '//*[@id="refresh-personalized-products" and @type="button" and @hidden]'
        ))
        self.assertTrue(self.document.xpath(
            '//*[@id="personalized-meta" and @hidden]'
        ))
        self.assertTrue(self.document.xpath(
            '//*[@id="personalized-summary" and @role="status" '
            'and @aria-live="polite"]'
        ))
        self.assertIn(
            "Öneriler bu cihazdaki son gezilenlere, kaydedilen kombinlere ve "
            "beden tercihlerinin stok durumuna göre hazırlanır; sunucuda "
            "kalıcı profil tutulmaz.",
            " ".join(self.document.xpath(
                '//*[@id="personalized-meta"]//text()'
            )).strip(),
        )

    def test_personalized_request_is_lazy_and_kept_separate_from_search(self):
        self.assertEqual(self.source.count("fetch('/personalized-recommendations'"), 1)
        self.assertIn("const personalizedCache = new Map()", self.source)
        self.assertIn("let personalizedController = null", self.source)
        self.assertIn("let personalizedRequestId = 0", self.source)
        self.assertIn("function ensurePersonalizedProducts()", self.source)
        self.assertIn("if (personalizedIsActive) ensurePersonalizedProducts()", self.source)
        self.assertIn(
            "personalizedProductsTab.addEventListener('click', () => "
            "setProductView('personalized'))",
            self.source,
        )
        self.assertNotRegex(
            self.source,
            r"localStorage\.(?:getItem|setItem|removeItem)\([^\n]*personalized",
        )

    def test_personalized_request_uses_the_bounded_profile_contract(self):
        self.assertIn("const PERSONALIZED_LIMIT = 12", self.source)
        self.assertIn("const PERSONALIZED_EXCLUDE_LIMIT = 100", self.source)
        self.assertIn(
            "recent_product_ids:profile.recent_product_ids.slice(0, RECENT_LIMIT)",
            self.source,
        )
        self.assertIn(
            "saved_outfits:profile.saved_outfits.slice(0, SAVED_OUTFITS_LIMIT)",
            self.source,
        )
        self.assertIn("size_preferences:profile.size_preferences", self.source)
        self.assertIn("exclude_product_ids:personalizedExcludedProductIds(profile, rotate)", self.source)
        self.assertIn("limit:PERSONALIZED_LIMIT", self.source)
        self.assertIn("source_product_id:sourceId", self.source)
        self.assertIn("recommended_product_id:recommendedId", self.source)
        for key in ("alpha", "numeric", "shoe", "jean_waist", "jean_length", "child"):
            self.assertIn(f"{key}:", self.source)

    def test_personalized_results_are_race_safe_render_reasons_and_rotate(self):
        self.assertIn(
            "controller.signal.aborted || requestId !== personalizedRequestId",
            self.source,
        )
        self.assertIn("if (personalizedFingerprint() !== fingerprint) return", self.source)
        self.assertIn("if (activeProductView === 'personalized') renderPersonalizedView()", self.source)
        self.assertIn(
            "renderProductGrid(personalizedProducts, { showReason:true })",
            self.source,
        )
        self.assertIn("productsById.set(String(product.id), product)", self.source)
        self.assertIn(
            "loadPersonalizedProducts({ force:true, rotate:true })",
            self.source,
        )
        self.assertIn(
            "ids.push(...personalizedProducts.map(product => personalizedProductId(product?.id)))",
            self.source,
        )
        self.assertIn("personalizedProducts = previousProducts", self.source)
        self.assertIn("Mevcut öneriler korunuyor.", self.source)
        self.assertIn(
            "const sameProfile = personalizedProfileFingerprint === fingerprint",
            self.source,
        )
        self.assertIn(
            "const previousProducts = sameProfile ? personalizedProducts.slice() : []",
            self.source,
        )
        self.assertIn("if (!sameProfile) {", self.source)
        self.assertIn(
            "const profileChangedDuringRequest = personalizedFingerprint() !== fingerprint",
            self.source,
        )
        self.assertIn("if (profileChangedDuringRequest) {", self.source)
        self.assertIn(
            "if (activeProductView === 'personalized') ensurePersonalizedProducts()",
            self.source,
        )

    def test_personalized_empty_error_and_loading_states_are_honest(self):
        self.assertIn("grid.setAttribute('aria-busy', 'true')", self.source)
        self.assertIn("Henüz kişisel öneri oluşturacak bir gezinme veya kayıt yok.", self.source)
        self.assertIn("Seni biraz daha tanımamız gerekiyor", self.source)
        self.assertIn("{ label:'Ürün ara', onClick:focusPersonalizedSearch }", self.source)
        self.assertIn("{ label:'Kaydedilen kombinlere git'", self.source)
        self.assertIn("Öneriler yüklenemedi", self.source)
        self.assertIn("{ label:'Tekrar dene'", self.source)
        self.assertIn("refreshPersonalizedProductsButton.disabled = !personalizedHasSignals()", self.source)
        self.assertIn("error.status === 404", self.source)
        self.assertIn("Öneri servisi eski sürümle çalışıyor.", self.source)
        self.assertIn("Docker ve uygulama sunucusu açıkken tekrar dene.", self.source)

    def test_saved_outfits_have_a_view_remove_and_clear_flow(self):
        self.assertTrue(self.document.xpath(
            '//*[@id="clear-saved-outfits" and @type="button" and @hidden]'
        ))
        self.assertIn("function renderSavedOutfits()", self.source)
        self.assertIn("createProductCard(outfit.source", self.source)
        self.assertIn("createProductCard(outfit.product", self.source)
        self.assertIn("function removeSavedOutfit(outfitId)", self.source)
        self.assertIn("button[data-remove-saved-outfit]", self.source)
        self.assertIn(
            "localStorage.removeItem(SAVED_OUTFITS_STORAGE_KEY)",
            self.source,
        )
        self.assertIn(
            "personalizedProductsTab,",
            self.source,
        )
        self.assertIn("setProductView('saved')", self.source)

    def test_feed_values_are_not_rendered_through_inner_html(self):
        self.assertNotIn("innerHTML", self.source)
        self.assertIn("textContent", self.source)
        self.assertIn("safeUrl", self.source)

    def test_static_ids_are_unique(self):
        ids = self.document.xpath("//@id")
        self.assertEqual(len(ids), len(set(ids)))

    def test_frontend_and_backend_supported_categories_stay_aligned(self):
        block = re.search(
            r"const COMBINABLE_CATEGORIES = new Set\(\[(.*?)\]\);",
            self.source,
            re.DOTALL,
        )
        self.assertIsNotNone(block)
        frontend_categories = set(re.findall(r"'([^']+)'", block.group(1)))
        backend_categories = set().union(
            TOP_CATEGORIES,
            BOTTOM_CATEGORIES,
            ONE_PIECE_CATEGORIES,
            LAYER_CATEGORIES,
        )
        self.assertEqual(frontend_categories, backend_categories)


if __name__ == "__main__":
    unittest.main()
