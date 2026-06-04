# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.4.0] - Unreleased

Workspace-wide minor bump (0.3.0 → 0.4.0) accompanying 11 sklearn-parity bug fixes surfaced by the new conformance test suite. All fixes change observable behaviour at the same hyperparameters, justifying a minor version increment.

### Added
- preprocess/quantile_transformer: QuantileTransformer REQ table (4 SHIPPED, 8 NOT-STARTED) — translating sklearn/preprocessing/_data.py (class QuantileTransformer :2540, _transform_col :2803-2866, _dense_fit :2679-2752) + nan_euclidean. SHIPPED: REQ-1 forward-transform value surface (uniform+normal, distinct+tied; references/landmarks numpy-faithful, fixed #1322), REQ-2 averaged forward+reversed interpolation (fixed #1321), REQ-3 Normal accuracy via Acklam ppf + clip (fixed #1320), REQ-5 error/parameter contracts. New design doc .design/preprocess/quantile_transformer.md. Consumer: re-export lib.rs. THREE-round critic-verified — 84-case stress matrix (n_quantiles {3,6,7,9,11,13} × distinct/tied/multi-plateau × uniform/normal, n_quantiles {<,=,>} n_samples, f32): uniform exact, normal within ~2.3e-9; 14 divergence green guards. Three divergences fixed (#1320 Normal ppf A&S→Acklam, #1321 plateau midpoint averaging, #1322 nanpercentile landmark *100/100 round-trip + linspace references). DIV-C (np.maximum.accumulate) found unobservable. NOT-STARTED: maximum.accumulate #1323, random subsample/random_state #1324, inverse_transform #1325, ignore_implicit_zeros/sparse #1326, quantile_transform free fn #1327, copy/fitted-attrs #1328, PyO3 #1329, ferray #1330. (#1319)
- preprocess/knn_imputer: KNNImputer REQ table (4 SHIPPED, 8 NOT-STARTED) — translating sklearn/impute/_knn.py (class KNNImputer) + sklearn/metrics/pairwise.py nan_euclidean_distances. SHIPPED: REQ-1 KNN imputation value surface (scaled distance + neighbor avg + column-mean fallback), REQ-2 nan_euclidean scaling (fixed #1305), REQ-3 empty-donor→column mean (fixed #1306), REQ-9 error/clamp contracts (n_neighbors>n_samples clamps not errors, fixed #1307). New design doc .design/preprocess/knn_imputer.md. Consumer: re-export lib.rs:131. THREE-round critic-verified — 33 oracle green guards (8×5/10×6 full matrices k∈{2,5} uniform+distance, mixed finite/inf donors, column-mean, clamp, f32) value-match live sklearn within ~1e-9. Five divergences fixed this iteration (#1305 distance scaling, #1306 column-mean, #1307 clamp, #1308 exact-match weighting, #1309 inf-distance donor inclusion). DIV-6 (#1310) exact-distance-tie donor selection is a documented carve-out (numpy argpartition unspecified tie order + ULP noise; not a meaningful parity target). NOT-STARTED: valid_mask column-drop #1311, missing_values #1312, add_indicator #1313, keep_empty_features #1314, callable weights/metric #1315, _BaseImputer surface #1316, PyO3 #1317, ferray #1318. (#1304)
- preprocess/rfe: RFE + RFECV REQ table (4 SHIPPED, 8 NOT-STARTED) — translating sklearn/feature_selection/_rfe.py (class RFE, class RFECV). SHIPPED: REQ-1 RFE ranking/support/elimination given static importances (matches sklearn _fit :337,:345-346 — sort-ascending + threshold removal + ranking accumulation), REQ-2 transform + error contracts, REQ-4 n_features_to_select>n_features keep-all (fixed #1296), REQ-9 RFECV optimal-count argmax given static cv_scores. New design doc .design/preprocess/rfe.md. Consumer: re-export lib.rs. Two-round critic-verified (15 oracle green guards incl. stable-importance ranking via coef_**2, RFECV tie-break, clamp boundary stress). HONEST: ferrolearn takes a STATIC importance vector (RFE) / pre-computed CV scores (RFECV); sklearn wraps an estimator + re-fits per round with squared importances + runs CV internally — the ranking/selection SHAPE matches, the estimator/re-fit/CV machinery is absent. NOT-STARTED: estimator+refit+squaring #1295, n_features_to_select=None/float #1297, float step #1298, importance_getter/verbose #1299, RFECV internal CV #1300, SelectorMixin surface #1301, PyO3 #1302, ferray #1303. (#1294)
- preprocess/sequential_feature_selector: SequentialFeatureSelector REQ table (3 SHIPPED, 8 NOT-STARTED) — translating sklearn/feature_selection/_sequential.py (class SequentialFeatureSelector). SHIPPED: REQ-1 greedy forward/backward search + lowest-index tie-break (matches sklearn _get_best_new_feature_score + max :280-294), REQ-2 error contracts, REQ-8 n_features_to_select<n_features + ensure_min_features=2 validation (fixed #1284/#1285). New design doc .design/preprocess/sequential_feature_selector.md. Consumer: re-export lib.rs:156-157. Two-round critic-verified (20 oracle green guards). HONEST: ferrolearn scores subsets via a USER CALLBACK; sklearn uses a wrapped estimator + cross_val_score — the greedy SHAPE matches but the estimator/CV scoring is absent. NOT-STARTED: estimator+cross_val_score #1286, n_features_to_select='auto' #1287, tol early-stop #1288, float fraction #1289, cv/scoring/n_jobs #1290, SelectorMixin surface #1291, PyO3 #1292, ferray #1293. (#1283)
- preprocess/select_percentile: SelectPercentile REQ table (3 SHIPPED, 7 NOT-STARTED) — translating sklearn/feature_selection/_univariate_selection.py (class SelectPercentile :589, _get_support_mask :669-686, f_classif :127). SHIPPED: REQ-1 ANOVA F-score value-match (anova_f_scores == f_classif on finite scores, critic-verified ~1e-6 to live sklearn), REQ-2 selection mask (fixed #1274), REQ-3 InsufficientSamples/ShapeMismatch/InvalidParameter error contracts. New design doc .design/preprocess/select_percentile.md. Consumer: re-export lib.rs:136. Two-round critic-verified (12 oracle green guards incl. 8-feature non-round percentiles 25/33/75/90, exact int()-floor ascending tie-fill, both numpy_percentile interpolation branches, f32 path). HONEST: f_classif-only score func, no _clean_nans/NaN handling, no SelectorMixin/pvalues_/fractional percentile, no PyO3. NOTE: SelectFpr/Fdr/Fwe live in sibling stat_selectors.rs. NOT-STARTED: _clean_nans #1275, pluggable score_func #1276, SelectorMixin surface #1277, pvalues_ #1278, fractional percentile #1279, PyO3 #1280, ferray #1281. (#1273)
- preprocess/target_encoder: TargetEncoder REQ table (3 SHIPPED, 10 NOT-STARTED) — translating sklearn/preprocessing/_target_encoder.py (class TargetEncoder) + _target_encoder_fast.pyx. SHIPPED: REQ-1 manual-smooth m-estimate value-match on fit().transform() (f64 bit-exact after #1261/#1262 fixes), REQ-2 unseen category → target_mean_ (global mean), REQ-3 InsufficientSamples/ShapeMismatch/InvalidParameter error contracts. New design doc .design/preprocess/target_encoder.md. Consumer: re-export lib.rs:138. Two-round critic-verified (17 oracle green guards: Probe-1 value match, multi-feature full matrix, binary-target, count-1 edge, pairwise across n<8/=8/13/101/=128/300/1000, interleaved-order accumulation). HONEST verify-and-document unit: manual-smooth f64 path bit-exact; structural gaps remain. NOT-STARTED: smooth='auto' empirical-Bayes default #1264, cross-fitting fit_transform #1265, target_type binary/multiclass #1266, categories param #1267, cv/shuffle/random_state #1268, string categories #1269, get_feature_names_out #1270, PyO3 #1271, ferray #1272, f32 lower-precision-than-f64 accumulation #1263. (#1260)
- preprocess/robust_scaler: RobustScaler REQ table (4 SHIPPED, 11 NOT-STARTED) — translating sklearn/preprocessing/_data.py (class RobustScaler :1445, robust_scale :1719). SHIPPED: REQ-1 per-column median/IQR value-match (non-constant; quantile_sorted linear-interp == numpy nanpercentile, critic-verified ~1e-12 to live sklearn center_/scale_), REQ-2 zero-IQR/constant column → 0 (fixed #1248), REQ-3 InsufficientSamples/ShapeMismatch error contracts, REQ-11 PyO3 _RsRobustScaler binding (extras.rs:1163, lib.rs:83). New design doc .design/preprocess/robust_scaler.md. Consumers: re-export lib.rs:124, PipelineTransformer, PyO3. Two-round critic-verified (fresh-oracle re-audit CLEAN incl. mixed constant/zero-IQR/normal matrix, non-zero-median constant, f32 path). HONEST: hardcoded 25/75 quantiles, always center+scale, dense-only, no inverse_transform/robust_scale free fn, NaN-poisons (no nanmedian). NOT-STARTED: quantile_range #1249, with_centering/with_scaling #1250, unit_variance #1251, inverse_transform #1252, robust_scale free fn #1253, NaN tolerance #1254, center_/scale_ attr names #1255, copy #1256, sparse CSC/CSR #1257, get_feature_names_out #1258, ferray #1259. (#1247)
- preprocess/label_binarizer: LabelBinarizer REQ table (7 SHIPPED, 5 NOT-STARTED) — translating sklearn/preprocessing/_label.py (class LabelBinarizer :180, label_binarize :430). SHIPPED: REQ-1 fit→sorted-unique classes_ (usize), REQ-2 multiclass one-hot values, REQ-3 binary single-column (pos_label on 2nd class), REQ-4 transform unknown-label ignore (fixed #1239), REQ-5 single-class all-zero column (fixed #1240), REQ-6 inverse strict >0.5 threshold (fixed #1241), REQ-7 multiclass inverse argmax — 7 oracle green guards in tests/divergence_label_binarizer.rs, two-round critic-verified (fresh-oracle re-audit CLEAN incl. interleaved/fully-unseen labels, single-class non-zero label, inverse at 0.5/0.500001/0.499999, k==1 inverse path matches sklearn _inverse_binarize_thresholding len==1→repeat). New design doc .design/preprocess/label_binarizer.md. Consumer: re-export lib.rs. HONEST: usize-only labels, dense not CSR, hardcoded neg_label=0/pos_label=1, no label_binarize free fn, no PyO3. NOT-STARTED: neg_label/pos_label params #1242, sparse_output CSR #1243, label_binarize free fn #1244, arbitrary label types+multilabel input #1245, PyO3 #1246. (#1238)
- preprocess/multi_label_binarizer: MultiLabelBinarizer REQ table (4 SHIPPED, 5 NOT-STARTED) — translating sklearn/preprocessing/_label.py (class MultiLabelBinarizer :688). SHIPPED: REQ-1 fit→sorted-unique classes_ (usize path, value-match to live sklearn), REQ-2 transform→dense multi-hot indicator (known labels, value-match), REQ-3 transform unknown-label ignore (fixed #1230), REQ-4 inverse_transform 0/1 validation (fixed #1231) — 6 oracle green guards in tests/divergence_multi_label_binarizer.rs, two-round critic-verified (fresh-oracle re-audit CLEAN incl. interleaved unknowns, all-unknown sample, 0.999/2.0/-1.0 inverse-reject, non-contiguous round-trip). New design doc .design/preprocess/multi_label_binarizer.md. Consumer: re-export lib.rs:155. HONEST: usize-only labels (no string/tuple), dense not CSR, inverse returns Vec<Vec<usize>> not tuples (faithful), no PyO3. NOT-STARTED: classes ctor param #1232, sparse_output CSR #1233, arbitrary orderable+hashable labels+object dtype #1234, optimized fit_transform #1235, PyO3 #1236; empty-y fit edge #1237. (#1229)
- preprocess/count_vectorizer: CountVectorizer REQ table (4 SHIPPED, 11 NOT-STARTED) — translating sklearn/feature_extraction/text.py (class CountVectorizer :929). SHIPPED: REQ-1 default fit/transform sorted-vocab count matrix (dense, critic-verified value-match to live sklearn), REQ-2 default token_pattern (drop length-1 tokens, `_` is a word char — fixed #1217), REQ-3 binary clipping, REQ-4 lowercase toggle, REQ-7 max_features top-N by corpus freq (scoped) — 8 oracle green guards in tests/divergence_count_vectorizer.rs (incl. the two #1217 tokenizer guards + the #1218 max_df guard), two-round critic-verified (fresh-oracle re-audit CLEAN incl. Unicode `café déjà`, interior `a1_b2`, max_df=1.0 boundary). Finalized design doc .design/preprocess/count_vectorizer.md (pre-staged iter 135). Consumer: re-export lib.rs:141. HONEST: output is DENSE Array2<f64> not CSR (REQ-11); no ngram/stop_words/analyzer/vocabulary/HashingVectorizer/PyO3. NOT-STARTED: max_df/min_df int-vs-float duality #1219, ngram_range #1220, analyzer hooks #1221, stop_words #1222, fixed vocabulary+dtype #1223, sparse CSR #1224, get_feature_names_out #1225, HashingVectorizer #1226, full ctor+_parameter_constraints+empty-vocab error #1227, PyO3 #1228. (#1216)
- preprocess/tfidf: TfidfTransformer REQ table (7 SHIPPED, 5 NOT-STARTED) — translating sklearn/feature_extraction/text.py (class TfidfTransformer :1483). SHIPPED: REQ-1 default smooth idf_, REQ-2 default transform (idf×tf + l2), REQ-3 smooth_idf=False, REQ-4 norm l1/l2/None, REQ-5 sublinear_tf, REQ-6 use_idf=False, REQ-7 idf_ accessor — full numeric contract critic-verified bit-identical to live sklearn (11 oracle green guards in tests/divergence_tfidf.rs incl. the full default-l2 value vector [[0.50854232,0.86103700,0],...]). Design doc .design/preprocess/tfidf.md (finalized; pre-staged iter 135). Consumer: re-export lib.rs:142. DEVIATION (R-DEV-4): smooth_idf=False df==0 edge → ferrolearn 1.0 vs sklearn inf (avoids 0*inf=nan footgun; CountVectorizer never emits all-zero columns). NOT-STARTED: sparse CSR #1211, _parameter_constraints #1212, TfidfVectorizer #1213, PyO3 #1214, ferray #1215. (#1210)
- preprocess/max_abs_scaler: MaxAbsScaler REQ table (4 SHIPPED, 8 NOT-STARTED) — translating sklearn/preprocessing/_data.py (class MaxAbsScaler :1116, maxabs_scale :1351). SHIPPED: REQ-1 per-column max-abs value-match (critic-verified bit-identical to live sklearn), REQ-2 zero-max_abs column→identity (MATCHES sklearn — unlike Min/StandardScaler, a zero-max_abs column is all-zero so x/scale_(1)=x equals leave-unchanged, oracle-confirmed via fit([[0],[0]]).transform([[5]])==[[5.0]]), REQ-3 inverse_transform round-trip, REQ-4 PyO3 _RsMaxAbsScaler fit/transform (maturin smoke test). New design doc .design/preprocess/max_abs_scaler.md. HONEST (verify-and-document): no fixable divergence found — 10 oracle green guards. NOT-STARTED: NaN-allow #1202, scale_/n_samples_seen_ #1203, partial_fit #1204, maxabs_scale fn #1205, copy #1206, sparse #1207, get_feature_names_out #1208, ferray #1209. (#1201)
- preprocess/standard_scaler: StandardScaler REQ table (4 SHIPPED, 9 NOT-STARTED) — translating sklearn/preprocessing/_data.py (class StandardScaler :696, scale :133). SHIPPED: REQ-1 per-column standardize value-match (critic-verified bit-identical to live sklearn, population std ddof=0), REQ-2 constant-column→0 (see Changed), REQ-3 inverse_transform round-trip, REQ-4 PyO3 _RsStandardScaler fit/transform/inverse_transform (maturin smoke test). New design doc .design/preprocess/standard_scaler.md. Consumers: PyO3 + PipelineTransformer + re-export. NOT-STARTED: var_/scale_/n_samples_seen_ #1192, with_mean/with_std/copy #1193, NaN-allow #1194, scale fn #1195, partial_fit #1196, sample_weight #1197, sparse #1198, get_feature_names_out #1199, ferray #1200. (#1190)
- preprocess/polynomial_features: PolynomialFeatures REQ table (2 SHIPPED, 9 NOT-STARTED) — translating sklearn/preprocessing/_polynomial.py (class PolynomialFeatures :99). SHIPPED: REQ-1 int-degree dense polynomial VALUES + exact column ORDER (== sklearn _combinations itertools order, critic-verified bit-identical incl. 3-feature/interaction_only layouts, 8 oracle green guards), REQ-8 transform input-validation matching sklearn check_array (see Changed). New design doc .design/preprocess/polynomial_features.md. Consumer: PipelineTransformer + re-export. NOT-STARTED: degree-tuple/min_degree #1181, order #1182, stateful fit/n_features_in_ #1183, powers_ #1184, get_feature_names_out #1185, sparse #1186, ctor/_parameter_constraints #1187, PyO3 #1188, ferray #1189. (#1179)
- preprocess/min_max_scaler: MinMaxScaler REQ table (4 SHIPPED, 8 NOT-STARTED) — translating sklearn/preprocessing/_data.py (class MinMaxScaler :291, minmax_scale :589). SHIPPED: REQ-1 per-column min-max value-match (critic-verified bit-identical to live sklearn, default+custom range), REQ-2 constant-column→feature_range[0] (see Changed), REQ-3 feature_range validation, REQ-7 PyO3 _RsMinMaxScaler fit/transform (maturin smoke test). New design doc .design/preprocess/min_max_scaler.md. Consumers: PyO3 + PipelineTransformer + re-export. NOT-STARTED: NaN-allow #1171, scale_/min_ attrs #1172, inverse_transform #1173, partial_fit #1174, minmax_scale fn #1175, copy/clip #1176, get_feature_names_out #1177, ferray #1178. (#1169)
- preprocess/ordinal_encoder: OrdinalEncoder REQ table (2 SHIPPED, 11 NOT-STARTED) — translating sklearn/preprocessing/_encoders.py (class OrdinalEncoder :1235). SHIPPED: REQ-1 string fit→sorted-unique categories_ (+ zero-row rejection), REQ-2 transform/fit_transform ordinal VALUES + unknown rejection (handle_unknown='error') — critic-verified bit-identical to live sklearn on the string path (7 oracle green guards in tests/divergence_ordinal_encoder.rs incl. ASCII+non-ASCII codepoint sort == np.unique, empty-fit-matches). New design doc .design/preprocess/ordinal_encoder.md. HONEST (verify-and-document): faithful String-only encoder; ordinal VALUES match sklearn but output dtype is usize vs sklearn float64 (R-DEV-3, coupled to absent NaN-sentinel features). NOT-STARTED: dtype #1158, numeric/mixed input #1159, use_encoded_value #1160, encoded_missing_value #1161, categories param #1162, infrequent #1163, inverse_transform #1164, get_feature_names_out #1165, ctor #1166, PyO3 #1167, ferray #1168. (#1157)
- preprocess/one_hot_encoder: OneHotEncoder REQ table (1 SHIPPED scoped, 8 NOT-STARTED) — translating sklearn/preprocessing/_encoders.py (class OneHotEncoder :458). SHIPPED: REQ-1 dense one-hot of contiguous 0..max integer columns, critic-verified bit-identical to live sklearn sparse_output=False (6 oracle green guards in tests/divergence_one_hot_encoder.rs incl. multi-column layout + column-order). New design doc .design/preprocess/one_hot_encoder.md. HONEST (verify-and-document): ferrolearn uses n_categories[j]=max(col)+1 (contiguous-integer assumption), NOT sklearn's categories_=_unique sorted-set — diverges structurally on non-contiguous integers ([2,5,9]: 10 cols vs sklearn 3) + cannot represent strings/floats; dense not sparse-default. NOT-STARTED: sparse #1149, categories_ unique-set #1150, handle_unknown #1151, drop/infrequent #1152, inverse_transform/get_feature_names_out #1153, ctor/_parameter_constraints #1154, PyO3 #1155, ferray #1156. (#1148)
- preprocess/normalizer: Normalizer REQ table (2 SHIPPED, 7 NOT-STARTED) — translating sklearn/preprocessing/_data.py (class Normalizer :1980, normalize :1866). SHIPPED: REQ-1 row-wise L1/L2/Max transform (zero-norm row unchanged), critic-verified bit-identical to live sklearn (5 oracle green guards incl. f32); REQ-2 transform input-validation matching sklearn check_array (see Changed). New design doc .design/preprocess/normalizer.md. Real consumers: PipelineTransformer impl + re-export lib.rs:119. NOT-STARTED: fit/param-constraints #1141, normalize free fn (axis/return_norm) #1142, copy #1143, n_features_in_/feature names #1144, sparse #1145, PyO3 #1146, ferray #1147. (#1139)
- preprocess/label_encoder: LabelEncoder REQ table (4 SHIPPED, 4 NOT-STARTED) — translating sklearn/preprocessing/_label.py (class LabelEncoder :34). SHIPPED: REQ-1 string fit→sorted-unique classes_, REQ-2 inverse_transform, REQ-3 transform/fit_transform, REQ-5 empty-fit parity (see Changed) — all critic-verified bit-identical to live sklearn on the string path (oracle green guards in tests/divergence_label_encoder.rs incl. np.unique sort-order match). New design doc .design/preprocess/label_encoder.md. HONEST: Array1<String>-only (sklearn accepts any dtype). NOT-STARTED: numeric/generic dtype #1135, error-contract message/NotFittedError #1136, PyO3 #1137, ferray #1138. (#1133)
- preprocess/binarizer: Binarizer REQ table (2 SHIPPED, 7 NOT-STARTED) — translating sklearn/preprocessing/_data.py (class Binarizer :2177, binarize :2120). SHIPPED: REQ-1 dense strict-greater transform (x.mapv(v>threshold?1:0)), critic-verified bit-identical to live sklearn (oracle green guards in tests/divergence_binarizer.rs); REQ-9 transform input-validation matching sklearn check_array (see Changed). New design doc .design/preprocess/binarizer.md. NOT-STARTED: copy #1126, fit/param-constraints #1127, binarize free fn #1128, n_features_in_/feature names #1129, sparse #1130, PyO3 #1131, ferray #1132. (#1122)
- preprocess/function_transformer: FunctionTransformer REQ table (1 SHIPPED scoped, 10 NOT-STARTED) — translating sklearn/preprocessing/_function_transformer.py (class FunctionTransformer). SHIPPED: element-wise forward Transform (x.mapv(scalar Fn(F)->F)), critic-verified bit-identical to live sklearn for ufuncs (5 oracle-grounded green guards in tests/divergence_function_transformer.rs: log1p/expm1/sqrt/log-NaN-inf/empty-shape). New design doc .design/preprocess/function_transformer.md. HONEST: ferrolearn's func is scalar Fn(F)->F (element-wise), NOT sklearn's whole-array func(X); no func=None default, inverse_transform, validate, check_inverse/fit, feature_names_out, kw_args, ctor surface, PyO3 binding, or ferray substrate. NOT-STARTED blockers #1112-#1121. (#1111)
- bayes/lib: ferrolearn-bayes crate-root RE-EXPORT BOUNDARY REQ table (2 SHIPPED, 1 NOT-STARTED) — mirrors sklearn/naive_bayes.py __all__ (:30-36, the 5 NB variants) + _BaseNB.predict_log_proba = jll−logsumexp(jll,axis=1) (:105-126). SHIPPED: re-export boundary (meta-crate + PyO3 RsGaussianNB/RsMultinomialNB/RsBernoulliNB/RsComplementNB consumers); log_softmax_rows == the numerically-stable jll−logsumexp normalization behind every Fitted*NB predict_log_proba/predict_proba, critic-verified CLEAN vs live oracle (4 green guards in tests/divergence_lib.rs: GaussianNB end-to-end ~1e-12, all-−inf row→NaN matching scipy, single-col→0.0, large-magnitude no-overflow). New design doc .design/bayes/lib.md; route repointed base.md→lib.md. NOT-STARTED: ferray substrate #1110. (#1108)
- linear/lib: ferrolearn-linear crate-root RE-EXPORT BOUNDARY REQ table (5 SHIPPED, 2 NOT-STARTED) — mirrors sklearn/linear_model/__init__.py __all__ (:48-98) + the base.py score mixins (ClassifierMixin.score :738-764 → accuracy_score; RegressorMixin.score :805-849 → r2_score). SHIPPED: re-export boundary (meta-crate + PyO3 consumers), ClassifierScore==mean accuracy, RegressorScore==in-regime R² (oracle 0.9152…), the constant-y R² edge (#1104 fixed) and log_proba (#1105 fixed); score traits ship on the grandfathered re-export basis (goal.md S5). New design doc .design/linear/lib.md; route repointed sgd.md→lib.md. HONEST: no production .score() caller yet, sample_weight unsupported (#1106), helpers on ndarray not ferray (#1107). (#1103)
- cluster/gmm: GaussianMixture well-separated labels_/predict PARTITION (up to a label permutation) + predict_proba/transform row-stochastic CONTRACT + well-separated weights_/means_/covariances_ VALUE-match (~1e-15) + matching defaults (Full/max_iter=100/tol=1e-3/n_init=1) + absolute score/score_samples/lower_bound_/aic/bic VALUE-match live sklearn 1.5.2 (the Full/Tied log|Σ| double-count #1093 now fixed, see Changed) + thin PyO3 fit/predict marshalling (_RsGaussianMixture), verified by 6 green guards + binding smoke test; REQ table (6 SHIPPED, 8 NOT-STARTED), two-round critic-verified (incl. an adversarial anisotropic/differing-log|Σ| score probe confirming the fix is not a symmetric coincidence). HONEST: covariances_ is 2D (k·d,d) not sklearn's 3D (k,d,d); no precisions_/precisions_cholesky_/n_iter_/sample(); init is k-means++ SEEDING not sklearn's init_params='kmeans' full-KMeans, so exact value parity off the well-separated regime DIVERGES (numpy-RNG-coupled); the binding is THIN (fit/predict only). NOT-STARTED: covariances_ shape #1094, precisions_ #1095, n_iter_ #1096, sample() #1097, init_params/reg_covar/*_init/warm_start #1098, off-separated value parity #1099, full binding surface #1100, ferray #1101. Completes ferrolearn-cluster. (#1092)
- cluster/optics: OPTICS core_distances_/ordering_/reachability_ VALUE-match live sklearn 1.5.2 (deterministic — genuine value-parity, incl. tie-prone fixtures) verified by 8 green guards; the traversal (#1080) now matches sklearn (see Changed); REQ table (3 SHIPPED, 10 NOT-STARTED), two-round critic-verified. HONEST: labels_ (Xi) still diverge on hard data (1500-pt: sklearn 21 clusters/1167 noise vs ferro 19/1228) — gated on the in-Xi min_cluster_size #1085 + cluster_hierarchy_ #1086; cluster_method='dbscan' #1084, predecessor_ -1-sentinel #1082, param surface #1088, validation ABI #1089, no PyO3 binding #1090, ferray #1091. (#1079)
- cluster/hdbscan: HDBSCAN labels_ PARTITION + noise set VALUE-match live sklearn 1.5.2 EXACTLY on well-separated/dense/outlier fixtures (deterministic — genuine partition parity) + probabilities range/noise-0 contract + defaults, verified by 4 green guards; the core-distance index (#1070) now matches sklearn (see Changed); REQ table (4 SHIPPED, 9 NOT-STARTED), two-round critic-verified. HONEST: ferrolearn implements the correct algorithm SHAPE (core-dist → mutual-reachability → MST → condensed tree → EOM) but probabilities_ VALUES (GLOSH formula), exact label integers, cluster_selection_method='leaf', metric/alpha/algorithm, centroids_/medoids_, and the epsilon semantics DIVERGE; no PyO3 binding. NOTE: sklearn 1.5.2 HDBSCAN has NO cluster_persistence_ attribute (that's the contrib package). NOT-STARTED: probability GLOSH values #1069, label integers #1071, cluster_selection_method/allow_single/max_cluster #1072, metric/alpha/algorithm #1073, PyO3 binding #1074, centroids_/medoids_/n_features_in_ #1075, error ABI #1076, epsilon semantics #1077, ferray #1078. (#1068)
- cluster/bayesian_gmm: BayesianGaussianMixture API/output contracts (predict_proba row-stochastic, weights sum to 1, shapes) + matching defaults (covariance_type=Full/max_iter=100/tol=1e-3/dirichlet_process) verified vs live sklearn 1.5.2 by green guards; REQ table (2 SHIPPED, 11 NOT-STARTED), two-round critic-verified. HONEST (R-HONEST-3/4): ferrolearn's impl is a HEURISTIC plain ML-EM, NOT sklearn's variational Bayes — no digamma-weight E-step, no Wishart precisions (degrees_of_freedom_/mean_precision_/precisions_cholesky_), Full/Tied responsibilities use the covariance DIAGONAL only, lower_bound_ is a proxy. REQ-1 (partition matches sklearn) is NOT-STARTED #1067: ferrolearn lacks sklearn's DP component pruning (2-blob: sklearn prunes to 1 cluster, ferrolearn keeps 2) — a prior draft green-guard fabricated the 2-blob expected value (R-CHAR-3 violation) and was removed. NOT-STARTED: VB E/M algorithm #1057, full-cov Mahalanobis #1058, true ELBO #1059, Bayesian attrs #1060, priors/n_init #1061, init_params/ctor #1062, partition-vs-pruning #1067, PyO3 binding #1063, n_iter_ #1064, numpy-RNG #1065, ferray #1066. (#1056)
- cluster/mini_batch_kmeans: MiniBatchKMeans labels_ PARTITION (up to a label permutation, well-separated regime) + predict/transform CONTRACTS (predict==labels_ holds) + thin PyO3 binding (_RsMiniBatchKMeans: fit/predict/labels_) verified vs live sklearn 1.5.2 by 5 green guards + binding smoke test; the n_init default (#1047) now matches sklearn (see Changed); REQ table (5 SHIPPED, 8 NOT-STARTED), two-round critic-verified. HONEST: exact cluster_centers_/inertia_ VALUES + labels_ integers DIVERGE — ferrolearn runs n_init FULL per-center-LR runs with max-shift convergence, vs sklearn's single best-of-init-trials + EWA-inertia early stopping + low-count reassignment on the numpy RNG. NOT-STARTED: ctor surface/sample_weight/error-ABI #1048, numpy-RNG #1049, algorithm structure (init-trials/EWA/n_steps_) #1050, value parity #1051, partial_fit #1052, thin binding surface/dtype #1053, low-count reassignment #1054, ferray #1055. (#1046)
- cluster/kmeans: KMeans labels_ PARTITION (up to a label permutation, well-separated regime) + predict/transform CONTRACTS + PyO3 binding marshalling (_RsKMeans) verified vs live sklearn 1.5.2 by 4 green guards + binding smoke test; the labels_/inertia_↔cluster_centers_ consistency (#1037) and n_init default (#1045) now match sklearn (see Changed); REQ table (6 SHIPPED, 8 NOT-STARTED), two-round critic-verified. HONEST: exact cluster_centers_/inertia_ VALUES + labels_ integers + n_iter_ DIVERGE — blocked by numpy-RNG init parity #1039, the convergence criterion + relative tol #1036, and empty-cluster relocation #1040. NOT-STARTED: convergence/relative-tol #1036, init param #1038, numpy-RNG #1039, value parity #1040, ctor surface/sample_weight/error-ABI #1041, score/fit_transform #1042, binding n_init default/dtype #1043, ferray #1044. (#1035)
- cluster/bisecting_kmeans: BisectingKMeans labels_ PARTITION (up to a label permutation, well-separated regime) + transform distance-to-centers CONTRACT verified vs live sklearn 1.5.2 by 3 green guards; the bisecting_strategy default (#1025) and n_init default (#1026) now match sklearn (see Changed); REQ table (4 SHIPPED, 9 NOT-STARTED), two-round critic-verified. HONEST: cluster_centers_/inertia_ VALUES + labels_ integer numbering + predict semantics DIVERGE — blocked by numpy-RNG init parity #1030, mean-subtraction #1028, the inner Lloyd/tol algorithm, tree-DFS label order #1027, tree-descent predict #1029; no CPython binding. NOT-STARTED: centers/inertia values #1024, labels DFS numbering #1027, mean-subtraction #1028, tree-descent predict #1029, numpy-RNG parity #1030, ctor surface init/k-means++/tol/sample_weight/algorithm #1031, error ABI #1032, PyO3 binding #1033, ferray #1034. (#1023)
- cluster/label_spreading: LabelSpreading contiguous-label transduction PARTITION verified vs live sklearn 1.5.2 by 3 green guards; the alpha ∈ (0,1) open-interval validation (#1009) and the classes_/n_classes/label-VALUE mapping (#1011) now match sklearn (see Changed); REQ table (3 SHIPPED, 10 NOT-STARTED), two-round critic-verified. HONEST: label_distributions_ VALUES DIVERGE — ferrolearn's normalized-Laplacian degree excludes the RBF self-affinity, inits unlabeled rows uniform, row-normalizes every iteration, converges on L2-at-end, vs sklearn degree-incl-self + zero-init + no-per-iter-norm + L1-at-start; predict/predict_proba nearest-neighbor not kernel-weighted; no CPython binding. NOT-STARTED: label_distributions_ value #1010, convergence #1012, tol default #1013, predict kernel-weighted #1014, transduction_/classes_/n_iter_ attrs #1015, ConvergenceWarning #1016, KNN connectivity graph #1017, error ABI #1018, ferray #1019, PyO3 binding #1020. (#1008)
- cluster/label_propagation: LabelPropagation contiguous-label transduction PARTITION verified vs live sklearn 1.5.2 by 2 green guards, and the classes_/n_classes/label-VALUE mapping now matches sklearn (#999, see Changed); REQ table (2 SHIPPED, 10 NOT-STARTED), two-round critic-verified. HONEST: label_distributions_ VALUES DIVERGE — ferrolearn zeroes the RBF self-affinity diagonal + inits unlabeled rows uniform + converges on L2-at-end, vs sklearn rbf_kernel diagonal 1 + zero-init + L1-at-start; predict/predict_proba are nearest-neighbor not sklearn's kernel-weighted combination; no CPython binding. NOT-STARTED: label_distributions_ value #997, convergence criterion #998, tol default #1000, predict kernel-weighted #1001, transduction_/classes_/n_iter_ attrs #1002, ConvergenceWarning #1003, KNN connectivity graph #1004, error ABI #1005, PyO3 binding #1006, ferray #1007. (#996)
- cluster/mean_shift: MeanShift explicit-bandwidth labels_ PARTITION (up to a label permutation, well-separated regime) verified vs live sklearn 1.5.2 by 3 green guards, and estimate_bandwidth kNN VALUE (default path) now matches sklearn (#985, see Changed); REQ table (2 SHIPPED, 10 NOT-STARTED), critic-verified. HONEST: cluster_centers_ VALUES + labels_ INTEGERS + n_iter_ DIVERGE — ferrolearn averages each merged mode group in seed order, sklearn retains the highest-intensity converged mode sorted by intensity; no CPython binding. NOT-STARTED: cluster_centers_ value/intensity-order #984, labels_ integers #986, ctor surface seeds/bin_seeding/min_bin_freq/cluster_all/n_jobs+drop tol #987, cluster_all=False orphan -1 #988, bin_seeding/get_bin_seeds #989, stop-threshold 1e-3*bandwidth #990, n_iter_ semantics #991, error ABI #992, PyO3 binding #993, ferray #994. (#983)
- cluster/affinity_propagation: AffinityPropagation message-passing math (responsibility + availability updates algebraically equivalent to sklearn 1.5.2, numeric diff 0.0) + labels_ PARTITION (up to a label permutation, well-separated regime) + n_clusters verified by 3 green guards; REQ table (3 SHIPPED, 11 NOT-STARTED), two-round critic-verified. HONEST: exact labels_/cluster_centers_indices_/n_iter_ VALUE parity does NOT ship — blocked by degeneracy-noise injection #972, convergence-window criterion #973, exemplar-refinement pass #974; there is no CPython binding (no _RsAffinityPropagation) #978. NOT-STARTED: random_state+noise #972, convergence #973, refinement #974, predict #975, affinity precomputed #976, affinity_matrix_/cluster_centers_indices_ attrs #977, binding #978, non-convergence -1-labels-vs-Err #979, equal-similarities short-circuit #980, copy/verbose #981, ferray #982. (#970)
- cluster/agglomerative: AgglomerativeClustering labels_ PARTITION (up to a label permutation) + n_clusters_ + all four linkages (Ward/Complete/Average/Single) verified vs live sklearn 1.5.2 by 2 green guards, across real consumers (birch.rs, feature_agglomeration.rs, PyO3 _RsAgglomerativeClustering); REQ table (4 SHIPPED, 8 NOT-STARTED), two-round critic-verified. HONEST: ferrolearn builds a TRUNCATED children_ (length n_samples-n_clusters, reused-slot IDs) + numbers labels by ascending-slot HashMap relabel, NOT sklearn's full (n_samples-1,2) dendrogram + _hc_cut heap cut — so the absolute labels_ numbering and children_ format DIVERGE (shared root cause #938, also gating birch/feature_agglomeration). NOT-STARTED: children_ format + absolute labels_ #938, n_clusters=2 default/ABI #963, ensure_min_samples=2 #964 (coupled to birch single-subcluster consumer), metric/connectivity #965, distance_threshold/distances_ #966, n_leaves_/n_connected_components_/memory #967, ferray #968. (#962, guards #969)
- cluster/birch: Birch subcluster_centers_ VALUE (CF centroid LS/N, as a SET of rows) + labels_ PARTITION (global Agglomerative-Ward, up to a label permutation) + threshold criterion verified vs live sklearn 1.5.2 — but ONLY in the n_subclusters<=branching_factor regime (real PyO3 consumer _RsBirch); REQ table (3 SHIPPED, 8 NOT-STARTED). HONEST: ferrolearn's CF data structure is a flat Vec + merge-on-overflow capped at branching_factor, NOT a real CF-tree (_split_node) — on a 60-point stress fixture sklearn yields 37 leaf subclusters vs ferrolearn 5 (root structural divergence #954). NOT-STARTED: CF-tree #954, n_clusters=3 default #955, predict/transform #956, subcluster_labels_ #957, ConvergenceWarning #958, partial_fit #959, compute_labels/copy #960, ferray #961. (#953)
- cluster/dbscan: DBSCAN labels_ + core_sample_indices_ value-match sklearn 1.5.2 EXACTLY (incl shared-border tie-break + noise) on the Euclidean/no-sample_weight path, verified by 6 green guards + real PyO3 consumer (_RsDBSCAN); eps>0/min_samples>=1 validation. REQ table (4 SHIPPED, 7 NOT-STARTED), two-round critic-verified. Carve-out (#952): a neighbor edge whose true distance is within a ULP of eps can round to opposite sides of the boundary (ferrolearn sum-sq<=eps^2 vs sklearn euclidean_distances/tree rounding) — exact eps-boundary parity NOT-STARTED. NOT-STARTED: eps=0.5 default/ABI #946, sample_weight #947, metric/p #948, algorithm #949, components_ #950, ferray #951. (#945)
- cluster/feature_agglomeration: FeatureAgglomeration ensure_min_features=2 + n_clusters/n_features validation + transform shape + mean-pooling arithmetic (as unordered set) verified vs live sklearn 1.5.2; REQ table (1 SHIPPED, 10 NOT-STARTED). HONEST: groups features into the SAME partition as sklearn but assigns a PERMUTED label index (root cause in agglomerative.rs _hc_cut+searchsorted vs active-slot), so labels_ and transform COLUMN ORDER do NOT value-match — NOT-STARTED #938; inverse_transform #940, missing params #941, fitted attrs #942, PyO3+ferray #943. (#937)
- cluster/spectral: SpectralClustering gamma>=0/n_clusters>=1/insufficient-samples parameter-validation verified vs live sklearn 1.5.2; REQ table (1 SHIPPED, 10 NOT-STARTED). HONEST: simplified variant — RBF-only affinity, kmeans-only assign_labels, row-L2 normalize of normalized-adjacency top-k — whose embedding/labels do NOT match sklearn's spectral_embedding (Laplacian bottom-k scaled by 1/dd); core embedding+label parity NOT-STARTED (#929 + KMeans-parity #934), affinity modes #931, assign_labels #932, params/n_clusters=8 #933, PyO3+ferray #935. Also cleared 5 toolchain clippy lints (birch/mean_shift/optics) + seeded their design docs. (#928)
- bayes/categorical: CategoricalNB feature_log_prob_ smoothing log((N_cjk+alpha)/(N_c+alpha·K_j)) + jll + predict/predict_proba/predict_log_proba + min_categories/n_categories_ + class_prior length-only + force_alpha/fit_prior + score + partial_fit same-categories verified value-correct vs live sklearn 1.5.2 (delegates to BaseNB; impl PipelineEstimator consumer — no PyO3 binding yet); REQ table (7 SHIPPED, 4 NOT-STARTED — unseen-category + predict-path negative #920, accessors+PyO3 binding #923, sample_weight+partial_fit ext #924, ferray #925). Completes ferrolearn-bayes. (#919)
- bayes/complement: ComplementNB complement weights -log((comp_count+alpha)/(comp_total+alpha·nf)) + jll X@weights.T + predict/predict_proba/predict_log_proba (norm=False AND norm=True) + class_prior length-only + force_alpha/fit_prior + score + partial_fit same-classes + negative-feature reject verified value-correct vs live sklearn 1.5.2 (delegates to BaseNB; _RsComplementNB threads norm + pipeline consumers); REQ table (7 SHIPPED, 3 NOT-STARTED — sample_weight + partial_fit classes= #915, fitted accessors + PyO3 surface #916, ferray #917). (#913)
- bayes/bernoulli: BernoulliNB smoothing (N_cj+alpha)/(N_c+2alpha) + jll/predict/predict_proba/predict_log_proba + binarize threshold + class_prior length-only + force_alpha/fit_prior + score + partial_fit same-classes verified value-correct vs live sklearn 1.5.2 (delegates to BaseNB; _RsBernoulliNB + pipeline consumers); REQ table (7 SHIPPED, 3 NOT-STARTED — sample_weight + partial_fit classes= #908, fitted accessors + PyO3 surface #909, ferray #910). (#905)
- bayes/multinomial: MultinomialNB feature_log_prob_ smoothing + predict/predict_proba/predict_log_proba + empirical/uniform prior + class_prior length-only (matches sklearn discrete NB) + negative-feature reject + force_alpha/fit_prior + partial_fit same-classes verified value-correct vs live sklearn 1.5.2 (delegates to BaseNB; _RsMultinomialNB + pipeline consumers); REQ table (7 SHIPPED, 3 NOT-STARTED — sample_weight #901, fitted accessors + PyO3 surface #902, ferray #903). (#899)
- bayes/gaussian: GaussianNB epsilon_ (global per-feature variance, no floor; #891), priors validation (sum≈1 + non-negative; #893), and joint_log_likelihood + predict/predict_proba/predict_log_proba + theta_/log_prior/score verified value-correct vs live sklearn 1.5.2 (delegates to BaseNB; _RsGaussianNB + pipeline consumers); REQ table (4 SHIPPED, 5 NOT-STARTED — sample_weight #894, partial_fit epsilon-once #895, fitted accessors #896). (#892)
- bayes/base: real `_BaseNB`/`_BaseDiscreteNB` shared base (`ferrolearn-bayes/src/base.rs`) — `BaseNB<F>` trait (predict=argmax(jll)→classes_, predict_log_proba=jll−logsumexp, predict_proba=exp; sklearn/naive_bayes.py:103,123-126,144) + `check_alpha` (`_check_alpha` floor 1e-10, :604-626). All 5 NB variants delegate their predict pipeline to it (behavior-preserving); REQ table (4 SHIPPED, 3 NOT-STARTED) critic-verified. Route swap conjugate.rs→base.rs; conjugate.rs (no-analog Normal-Normal prior) now unrouted like umap.rs. GaussianNB epsilon_ value divergence found + tracked (#891). (#889)
- neighbors/radius_neighbors: RadiusNeighborsClassifier predict + tie-break + outlier_label, RadiusNeighborsRegressor predict, and radius_neighbors value/set verified value-correct vs live sklearn 1.5.2 (6 green guards + pipeline/graph.rs consumers); REQ table (3 SHIPPED, 11 NOT-STARTED — predict_proba/score test-only #887, outlier_label='most_frequent' #881, multi-output #884). Completes ferrolearn-neighbors. (#880)
- neighbors/knn: KNeighborsClassifier predict + smallest-label tie-break, KNeighborsRegressor predict (uniform+distance), kneighbors value, and HasClasses verified value-correct vs live sklearn 1.5.2 (7 green guards + ferrolearn-python/pipeline/graph.rs consumers); REQ table (5 SHIPPED, 7 NOT-STARTED — predict_proba/score value-correct but unbound #877, reg multi-output #875, missing params #876) (#873)
- neighbors/nearest_neighbors: NearestNeighbors kneighbors (explicit-X k-NN value), radius_neighbors (value/set), and query-time error guards verified value-correct vs live sklearn 1.5.2 (4 green guards + non-test consumers in graph.rs); REQ table (4 SHIPPED, 7 NOT-STARTED — X=None self-exclusion #866, radius sort_results default #867, metric/p/radius params #868) (#864)
- neighbors/balltree: BallTree query k-NN value, tie-SET, and within_radius value contracts verified value-correct vs live sklearn 1.5.2 (5 green guards + real non-test consumers in knn/nearest_neighbors/radius_neighbors); REQ table (3 SHIPPED, 9 NOT-STARTED — metric/kernel_density/two_point_correlation, k>n+empty-X ValueError #858 blocked on query->Result threading cf #831) (#854)
- neighbors/local_outlier_factor: LocalOutlierFactor contamination="auto" default (Contamination enum) + offset_ (-1.5/percentile) + negative_outlier_factor_ contract, predict via nof<offset_, decision_function=score_samples-offset_; REQ table (5 SHIPPED, 6 NOT-STARTED) critic-verified vs sklearn 1.5.2 (#844 #847 #848 #849)
- neighbors/nearest_centroid: shrink_threshold s+=median(s) + clamp removal, n_classes<2 + zero-variance ValueError; REQ table (#836 #837 #838 #839 #840)
- neighbors/kdtree: KDTree single-row k-NN query verified value-correct vs live sklearn (REQ table; k>n error contract NOT-STARTED #831, blocked on consumer Result-threading) (#830)
- neighbors/graph: kneighbors_graph + radius_neighbors_graph self-exclusion (include_self=False default, zero diagonal) + REQ table; cleared 2 crate clippy debts (#822 #823 #824)
- metrics/classification: log_loss eps + roc_curve drop_intermediate + det_curve endpoint + calibration searchsorted + top_k tie-break edge-parity fixes; REQ table (completes ferrolearn-metrics) (#806)
- metrics/pairwise: REQ table + 11 value-contract guards (present distance/kernel functions verified value-correct to ULP vs live sklearn) (#788)
- translate(svm): REQ-1/8 gamma='auto' (Gamma enum scale/auto/value) + shrinking/break_ties/default alignment (#641 partial)
- translate(linear_svc): REQ-6 multi_class {ovr, crammer_singer} + per-class coef pin (#623)
- translate(linear_svc): REQ-9 class_weight {None, balanced, dict} (#626)
- translate(linear_svc): REQ-8 dual {auto, True, False} + unsupported-combination rejects (#625)
- translate(linear_svc): REQ-5 penalty {l2, l1} + l1 solver + (l1,hinge) reject (#622)
- translate(linear_svc): REQ-11 n_iter_/n_features_in_ accessors + tol>0 validation (#627)
- translate(linear_svc): REQ-4 pin hinge-loss coef_/intercept_ vs live oracle (#621)
- translate(linear_svc): REQ-3 pin predict + classes_ vs live oracle (#620)
- translate(linear_svc): REQ-7 fit_intercept + intercept_scaling (penalized augmented column) (#624)
- translate(linear_svc): REQ-2 binary decision_function (n,) + oracle pin (#619)
- translate(linear_svc): REQ-1/10 CRUX — drop C/n scaling + liblinear dual CD + penalized augmented intercept (#618)
- translate(linear_svr): REQ-9 fitted-attr contract (length-1 intercept_, n_features_in_) + param validation (#614)
- translate(linear_svr): REQ-6 dual param (auto/True/False) (#612)
- translate(linear_svr): REQ-8 n_iter_ + ConvergenceWarning at max_iter (#613)
- translate(linear_svr): REQ-2 pin predict vs oracle (gated on #607) (#608)
- translate(linear_svr): REQ-4 pin squared_epsilon_insensitive vs oracle (#610)
- translate(linear_svr): REQ-5 fit_intercept + intercept_scaling (penalized augmented column) (#611)
- translate(linear_svr): REQ-3 epsilon default 0.1 -> 0.0 (#609)
- translate(linear_svr): REQ-1/7 CRUX — drop C/n scaling to plain C + convergent solver (#607)
- translate(lda): REQ-10 eigen solver (generalized eigh(Sb,Sw)) (#596)
- Translation unit: ferrolearn-linear/lda.rs — eigen solver (#596) (#605)
- translate(lda): REQ-11 shrinkage (None/auto Ledoit-Wolf/float) (#597)
- translate(lda): REQ-9 lsqr solver (#595)
- Translation unit: ferrolearn-linear/src/lda.rs — lsqr solver + shrinkage (#604)
- translate(lda): REQ-15 tol rank thresholds (#601)
- translate(lda): REQ-12 store_covariance + covariance_ (#598)
- translate(lda): REQ-7 priors (None=empirical + provided) (#593)
- translate(lda): REQ-4 predict_log_proba + smallest_normal floor (#591)
- translate(lda): REQ-3 prior-aware predict_proba + register LDA in binding (#590)
- translate(lda): REQ-13 explained_variance_ratio_ oracle pin (#599)
- translate(lda): REQ-8 coef_/intercept_/xbar_ fitted attrs (#594)
- translate(lda): REQ-5 transform (X-xbar_)@scalings_ parity (#592)
- translate(lda): REQ-2 predict argmax (imbalanced-prior label pin) (#589)
- translate(lda): REQ-1 svd solver + decision_function parity (affine X@coef.T+intercept) (#588)
- translate(qda): REQ-4 pin predict_log_proba + smallest_normal floor + expose (#578)
- translate(qda): REQ-9 store_covariance + covariance_ accessor (#582)
- translate(qda): REQ-6 provided priors (None=empirical, array verbatim) (#580)
- translate(qda): REQ-11 expose means_/priors_/scalings_/rotations_/covariance_ (#584)
- translate(qda): REQ-10 tol + collinearity warning + SVD/pseudo-inverse for rank-deficient (#583)
- translate(qda): REQ-5 pin regularized decision vs Q(reg_param=0.5) (#579)
- translate(qda): REQ-3 pin predict_proba + expose on RsQDA (#577)
- translate(qda): REQ-2 pin predict label-for-label vs oracle (#576)
- translate(qda): REQ-1 pin decision_function vs live _decision_function (Cholesky-inv == SVD) (#575)
- translate(isotonic): REQ-10 free isotonic_regression() + check_increasing() (#571)
- translate(isotonic): REQ-9 expose X_min_/X_max_/X_thresholds_/y_thresholds_/increasing_ (#570)
- translate(isotonic): REQ-6 increasing='auto' via Spearman check_increasing (#567)
- translate(isotonic): REQ-5 y_min/y_max clipping of fitted range (#566)
- translate(isotonic): REQ-7 sample_weight weighted PAVA (#568)
- translate(isotonic): REQ-2 pin decreasing-PAVA pooled values vs oracle (#564)
- translate(isotonic): REQ-1 pin increasing-PAVA pooled y_thresholds_ vs oracle (#563)
- translate(isotonic): REQ-8 _make_unique weighted duplicate-X collapse (#569)
- translate(isotonic): REQ-4 default out_of_bounds Clip -> Nan (#565)
- translate(glm): REQ-11 warm_start (#557)
- translate(glm): REQ-11 warm_start (#557)
- translate(glm): REQ-10 solver param lbfgs/newton-cholesky + gradient-norm stop (#556)
- translate(glm): REQ-13 score(X,y)=D2 deviance score (#559)
- translate(glm): REQ-14 expose n_iter_ + per-family y-domain validation (#560)
- translate(glm): REQ-12 sample_weight (#558)
- translate(glm): REQ-3 pin Tweedie(power) vs oracle for log-link powers (#550)
- translate(glm): REQ-2 pin Gamma vs oracle + reject y<=0 domain (#549)
- translate(glm): REQ-1 pin Poisson coef_/intercept_ vs live oracle (#548)
- translate(glm): REQ-5 intercept init = link(weighted_mean(y)) (#552)
- translate(glm): REQ-9 TweedieRegressor default power 1.5 -> 0.0 (#555)
- translate(glm): REQ-8 TweedieRegressor link param auto/identity/log (#554)
- translate(glm): REQ-7 predict applies link.inverse not unconditional exp (#553)
- translate(glm): REQ-4/6 CRUX objective — mean half-deviance + 0.5*alpha, intercept UNPENALIZED (#551)
- translate(sgd): REQ-13 early_stopping + validation_fraction + n_iter_no_change (#533)
- Translation unit: ferrolearn-linear/sgd.rs — early_stopping + validation_fraction (REQ-13) (#546)
- translate(sgd): REQ-19 anti-pattern cleanup — unreachable!()/unwrap in kernel (#537)
- translate(sgd): REQ-14 average / ASGD (#534)
- translate(sgd): REQ-15 class_weight + sample_weight (#535)
- translate(sgd): REQ-18 SGDOneClassSVM estimator missing (builder) (#536)
- translate(sgd): REQ-9b epsilon not validated to [0, inf) for Huber/EpsilonInsensitive/SquaredEpsilonInsensitive (#544)
- translate(sgd): REQ-11 fit_intercept flag (#531)
- translate(sgd): REQ-3 missing squared_epsilon_insensitive regressor loss (#524)
- translate(sgd): REQ-2 missing squared_hinge + perceptron classifier losses (#523)
- Translation unit: ferrolearn-linear/sgd.rs — squared_hinge/perceptron/squared_epsilon_insensitive losses (#543)
- translate(sgd): REQ-8 adaptive schedule — divisor 5 + n_iter_no_change/best_loss trigger (#528)
- translate(sgd): REQ-10 convergence — best_loss + n_iter_no_change + tol on sumloss + dloss clip (#530)
- Translation unit: ferrolearn-linear/src/sgd.rs — SGD convergence + adaptive epoch tail (#522 #530 #528) (#542)
- translate(sgd): REQ-12 shuffle flag (#532)
- Translation unit: ferrolearn-linear/sgd.rs shuffle flag (REQ-12 #532) (#541)
- translate(sgd): REQ-5 penalty l1/elasticnet + l1_ratio (truncated-gradient u/q) (#526)
- translate(sgd): REQ-5 penalty l1/elasticnet + l1_ratio via Tsuruoka truncated gradient (u/q cumulative penalty) (#526)
- translate(sgd): REQ-9 default params (classifier learning_rate=optimal/eta0=0.0/power_t=0.5; epsilon=0.1) (#529)
- translate(sgd): REQ-7 optimal schedule omits t0 (optimal_init) offset (#527)
- translate(sgd): REQ-4 L2 update — global wscale shrink-before-gradient, not inline per-feature (#525)
- translate(ransac): REQ-6 n_inliers_best=1 init and >= acceptance gate (#514)
- translate(ransac): REQ-5 refit-once-after-loop + inlier_mask_ from subset model (#513)
- translate(ransac): REQ-9 MAD-zero parity — remove 1e-6 substitution (#517)
- translate(ransac): REQ-4 selection criterion — rank by base-estimator R² score (n_inliers, score), not residual_sum (#512)
- Translation unit: ferrolearn-linear/src/quantile_regressor.rs (mirrors sklearn QuantileRegressor) (#505)
- Translation unit: ferrolearn-linear/src/quantile_regressor.rs — exact LP fit (#510)
- Translation unit: ferrolearn-linear/src/huber_regressor.rs (mirrors sklearn HuberRegressor) (#494)
- Translation unit: ferrolearn-linear/src/huber_regressor.rs — joint [coef,intercept,scale] L-BFGS Huber fit (#503)
- Translation unit: ferrolearn-linear/src/omp.rs (mirrors sklearn OrthogonalMatchingPursuit) (#487)
- Translation unit: ferrolearn-linear/src/lars.rs (mirrors sklearn Lars/LassoLars) (#481)
- Translation unit: ferrolearn-linear/src/ard.rs (mirrors sklearn ARDRegression) (#473)
- Translation unit: ferrolearn-linear/src/bayesian_ridge.rs (mirrors sklearn BayesianRidge) (#463)
- Translation unit: ferrolearn-linear/bayesian_ridge.rs — MacKay evidence-max fit (#472)
- Translation unit: ferrolearn-linear/src/logistic_regression_cv.rs (mirrors sklearn LogisticRegressionCV) (#455)
- Translation unit: ferrolearn-linear/src/logistic_regression.rs (mirrors sklearn LogisticRegression) (#441)
- Translation unit: ferrolearn-linear/src/elastic_net_cv.rs (mirrors sklearn ElasticNetCV) (#430)
- Translation unit: ferrolearn-linear/src/lasso_cv.rs (mirrors sklearn LassoCV) (#420)
- Translation unit: ferrolearn-linear/src/elastic_net.rs (mirrors sklearn ElasticNet) (#416)
- Translation unit: ferrolearn-linear/src/lasso.rs (mirrors sklearn/linear_model/_coordinate_descent.py Lasso) (#406)
- Translation unit: ferrolearn-linear/src/ridge_classifier.rs (mirrors sklearn RidgeClassifier) (#404)
- Translation unit: ferrolearn-linear/src/ridge_cv.rs (mirrors sklearn RidgeCV) (#402)
- Translation unit: ferrolearn-linear/src/ridge.rs (mirrors sklearn/linear_model/_ridge.py Ridge) (#383)
- Translation unit: ferrolearn-linear/src/linalg.rs — SVD min-norm lstsq (fixes #376/#377; mirror LAPACK gelsd) (#379)
- Translation unit: ferrolearn-linear/src/linear_regression.rs (mirrors sklearn/linear_model/_base.py LinearRegression) (#370)

- **Multi-output Ridge regression** in `ferrolearn-linear`. New `FittedRidgeMulti<F>` type plus `Fit<Array2<F>, Array2<F>> for Ridge<F>` impl share the existing single-output `Ridge`'s hyperparameter struct but solve for an `(n_features, n_targets)` coefficient matrix in a single shared Cholesky factorization of `X^T X + αI`. Backed by a new `cholesky_solve_multi` + `solve_ridge_multi` pair in `linalg.rs`; the factor cost is `O(p^3)` paid once regardless of `t`. Donated from `forecast-bio/decode`'s `forecast-decode-regression::ridge_multi` (the per-PC ridge fit in the DINOv3 decoding pipeline) where the multi-target path is the hot path.
- **`Powell` direction-set optimizer** in `ferrolearn-numerical::optimize`. Derivative-free ND minimization matching `scipy.optimize.minimize(method='powell')`. Builder API mirrors `NewtonCG` / `TrustRegionNCG` (`Powell::new().with_max_iter(...).with_ftol(...).minimize(f, x0)`), and reuses the existing `OptimizeResult` (gradient field is zero-filled since Powell is derivative-free). Donated from `forecast-bio/decode`'s `forecast-decode-motion::optimize::powell` where it lines up `(dy, dx, theta)` for FFT-seeded motion correction.

### Fixed (sklearn-parity bugs)

- **#334 LogisticRegression loss normalisation** — removed the `1/n` averaging in both the binary and multinomial branches so the loss has units of `sum`, matching sklearn's `J = C * sum + 0.5 * ||w||^2`. At the same `C`, effective regularization is now `n×` weaker than before (i.e. matches sklearn).
- **#335 MAPE convention** — `mean_absolute_percentage_error` now returns the fraction (no `×100`), matching sklearn. Public-API breaking change at the metric level.
- **#336 spd_inverse diagonal-only bug** — the Cholesky-based triangular-inverse loop in `ferrolearn-covariance` was reading uninitialised `l_inv` entries during forward substitution, producing a diagonal-only "inverse" that silently corrupted every `precision_` and `mahalanobis(...)` output. Rewrote the loop to iterate by column with the rows-already-known invariant.
- **#337 MinCovDet FastMCD post-processing** — added sklearn's consistency correction (`median(mahal^2) / chi2_quantile(0.5, p)`) and reweighting steps. Added an `invert_with_shrinkage` helper that detects rank-deficient support covariances via the Cholesky-pivot ratio and applies trace-relative Tikhonov shrinkage so distance computations remain stable even when the support set lands on a near-1D subspace.
- **#339 Lars equiangular path** — replaced the forward-stepwise (OLS-on-active-set) implementation with the true LARS algorithm per Efron, Hastie, Johnstone & Tibshirani (2004). New `lars_path` shared helper computes the equiangular direction via the `X_A^T X_A` solve and steps the size that brings one new feature to equal absolute correlation.
- **#340 QuantileRegressor IRLS** — replaced `w_prev = eps` initialisation with an OLS warm-start. The eps-initialisation made the L1 linearisation diagonal `scaled_alpha / eps` huge on iteration 1, forcing `w ≈ 0` and producing predictions 25× off.
- **#341 SVM `gamma="scale"` parity (partial)** — confirmed and documented that `RbfKernel::new()` returns `gamma=None` which the kernel silently treated as `gamma=1`, while sklearn's `gamma="scale"` (the SVM default since 0.22) computes `1 / (n_features * X.var())`. Conformance tests now use the explicit scale gamma; SVC, NuSVC, and OneClassSVM pass. SVR/NuSVR/LinearSVR retain a separate epsilon-tube SMO divergence as the remaining open scope.
- **#342 TruncatedSVD Bessel correction** — `explained_variance_` now divides by `n_samples` (ddof=0) to match sklearn's `np.var(X_transformed, axis=0)`. Older `n-1` divisor produced values off by `n/(n-1)`.
- **#343 GraphicalLasso alpha-on-diagonal** — the Friedman et al. 2008 algorithm initialises `W = S + alpha * I` for numerical stability; sklearn strips this `+alpha` from the diagonal at output. Added the matching trim step inside `solve_glasso` so `fitted.covariance()` matches sklearn (and not `S + alpha * I`).
- **#344 OrdinalEncoder category order** — categories are now sorted lexicographically during fit, matching sklearn's `OrdinalEncoder.categories_`. Earlier ferrolearn used first-seen order.
- **#345 VarianceThreshold strict comparison** — replaced naive two-pass variance with Welford's online algorithm so constant columns produce *exactly* zero variance, making `threshold=0.0` correctly drop zero-variance columns. The naive sum/n then sum((v-mean)²)/n was accumulating ~1e-34 FP noise that defeated the strict `>` comparison.

### Added — Comprehensive conformance coverage (#338, follow-up to #333)

Total: **156 conformance tests passing across 13 crates** with 25 ignored, each ignored entry annotated with a tracking issue. Surface coverage gates on 12 of 13 crates lock the gate so any new public estimator must be tested or explicitly excluded.

- **Wave 1 — linear gap fixtures + tests** (28 estimators): HuberRegressor, BayesianRidge, ARDRegression, QuantileRegressor, Lars, LassoLars, OrthogonalMatchingPursuit, RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV, LDA, QDA, RidgeClassifier, LinearSVC, LinearSVR, SVC, SVR, NuSVC, NuSVR, OneClassSVM, SGDClassifier, SGDRegressor, RANSACRegressor, IsotonicRegression, PoissonRegressor, GammaRegressor, TweedieRegressor.
- **Wave 2 — decomp gap fixtures + tests** (17 estimators): TruncatedSVD, FastICA, KernelPCA, FactorAnalysis, IncrementalPCA, SparsePCA, DictionaryLearning, MiniBatchNMF, LatentDirichletAllocation, CCA, PLSRegression, PLSCanonical, Isomap, MDS, LLE, SpectralEmbedding, t-SNE.
- **Wave 3 — tree gap fixtures + tests** (13 estimators): ExtraTreeClassifier/Regressor (single), ExtraTrees{Classifier,Regressor}, BaggingClassifier/Regressor, AdaBoostRegressor, HistGradientBoosting{Classifier,Regressor}, IsolationForest, RandomTreesEmbedding, VotingClassifier/Regressor.
- **Wave 4 — cluster + neighbors + bayes + neural + covariance gaps** (17 estimators): AffinityPropagation, BayesianGaussianMixture, BisectingKMeans, FeatureAgglomeration, HDBSCAN, LabelPropagation, LabelSpreading, LocalOutlierFactor, NearestCentroid, NearestNeighbors, RadiusNeighbors{Classifier,Regressor}, CategoricalNB, MLPRegressor, BernoulliRBM, GraphicalLasso, EllipticEnvelope.
- **Wave 5 — kernel gap fixtures + tests** (6 estimators): GaussianProcessRegressor, GaussianProcessClassifier, Nystroem, RBFSampler, KernelRidge (RBF), KernelRidge (polynomial).
- **Wave 6 — preprocess gap fixtures + tests** (13 utilities): OrdinalEncoder, LabelBinarizer, MultiLabelBinarizer, VarianceThreshold, SelectKBest, SelectPercentile, SelectFromModel (api-gap), RFE (api-gap), KNNImputer, SplineTransformer, GaussianRandomProjection, SparseRandomProjection, FunctionTransformer.
- **Wave 7 — model-sel gap fixtures + tests** (8 utilities): LeaveOneOut, LeavePOut, ShuffleSplit, GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, DummyClassifier, DummyRegressor.
- **Wave 8 — surface-coverage gates** in `tests/conformance_surface_coverage.rs` for 12 additional crates (ferrolearn-tree, cluster, decomp, preprocess, metrics, neighbors, bayes, model-sel, kernel, covariance, neural — numerical excluded as it has no `pub use` surface to inventory). Each crate ships `_surface_inventory.toml` listing every public symbol and `_surface_exclusions.toml` documenting items not yet covered with a `#338 follow-up` tag.

### Fixed — bugs surfaced by the comprehensive conformance suite

All listed are filed but not yet patched; the related conformance test is `#[ignore]`d with a pointer to the issue.

- **#334** `LogisticRegression` data-fit normalization mismatch (sklearn-parity, `1/n` vs `1` weighting at the same `C`).
- **#335** `mean_absolute_percentage_error` returns ×100 of sklearn's value.
- **#336** `spd_inverse()` in `ferrolearn-covariance` returns a diagonal matrix instead of the true inverse — silently corrupts `precision_` across the whole crate.
- **#337** `MinCovDet` FastMCD divergence beyond expected subset variance (investigation).
- **#339** `Lars` coefficient path diverges 2× from sklearn at the same `n_nonzero_coefs`.
- **#340** `QuantileRegressor` predictions diverge 25× from sklearn — IRLS does not reach sklearn's HiGHS optimum.
- **#341** SVM family (`LinearSVR`, `NuSVC`, `NuSVR`, `OneClassSVM`, `SVR`) wide divergence from sklearn's libsvm — gamma=scale + SMO/QP investigation.
- **#342** `TruncatedSVD.explained_variance_` uses Bessel correction (ddof=1) while sklearn uses ddof=0.
- **#343** `GraphicalLasso.covariance_` diagonal off by exactly `alpha` vs sklearn.
- **#344** `OrdinalEncoder` uses first-seen category order; sklearn uses lex.
- **#345** `VarianceThreshold(threshold=0.0)` does not drop zero-variance columns (strict-vs-non-strict comparison off by one).

### Added — Conformance test infrastructure (#333)

- **`ferrolearn-test-oracle` crate** — workspace-internal helper crate with:
  - Algorithm-class tolerance constants (`TOL_LINEAR_FIT_*`, `TOL_TREE_PRED_*`, `TOL_CLUSTER_CENTER_*`, `TOL_METRIC_*`, `TOL_COVARIANCE_*`, etc.) so tolerances are documented in one place rather than hardcoded per test.
  - Fixture loader (`load_fixture(name)`) that walks up to the workspace root and returns a typed `Fixture` with optional per-fixture `tolerance` override and `divergence_id` annotation.
  - Assertion helpers (`assert_close`, `assert_close_slice`, `assert_close_rows_sign_ambiguous` for PCA-style sign-ambiguous outputs, `assert_labels_equal`, `assert_ari_ge`).
  - Adjusted Rand Index implementation for label-permutation-invariant cluster comparison.
  - Toml parsers for `_divergences.toml`, `_surface_inventory.toml`, `_surface_exclusions.toml`.

- **Fixture schema v2** — `fixtures/README.md` documents backwards-compatible additions: optional `sklearn_pin`, `tolerance: { rel, abs }`, and `divergence_id` fields. All v1 fixtures continue to load.

- **`conformance_sklearn.rs` test files** in 13 crates, exercising 64 sklearn parity tests against the fixture corpus:
  - ferrolearn-linear (5 tests), ferrolearn-tree (7), ferrolearn-cluster (9), ferrolearn-decomp (2), ferrolearn-preprocess (13), ferrolearn-metrics (6), ferrolearn-neighbors (2), ferrolearn-bayes (5), ferrolearn-model-sel (3), ferrolearn-numerical (3), ferrolearn-kernel (1), ferrolearn-covariance (9), ferrolearn-neural (1).
  - 57 passing, 9 ignored with explicit annotations pointing to tracking issues or documented divergences.

- **7 new fixtures** for previously-untested estimators: `empirical_covariance`, `shrunk_covariance`, `ledoit_wolf`, `oas`, `min_cov_det`, `kernel_ridge`, `mlp_classifier`. Generator at `scripts/generate_gap_fixtures.py`.

- **`_divergences.toml` registries** in `ferrolearn-linear/`, `ferrolearn-cluster/`, `ferrolearn-bayes/`, `ferrolearn-covariance/` documenting 6 known-and-justified divergences from sklearn (coordinate-descent path differences, OPTICS xi-extraction variant, ComplementNB internal sign convention, OAS Chen-2010 vs sklearn-simplified formula, FastMCD subset-selection variance, L-BFGS path differences in LogisticRegression).

### Fixed (real bugs surfaced by the conformance suite)

- *(filed, not yet patched)* **#334** — `LogisticRegression` normalizes the data-fit term by `1/n` while sklearn does not, making ferrolearn's effective regularization `n×` stronger than sklearn's at the same `C`. Conformance test is `#[ignore]`d pending fix.
- *(filed, not yet patched)* **#335** — `mean_absolute_percentage_error` returns the value multiplied by 100 (percentage) while sklearn returns the unscaled fraction. Cross-library numerical traps for porters.
- *(filed, not yet patched)* **#336** — `spd_inverse()` in `ferrolearn-covariance` returns a diagonal matrix `diag(1/L[i,i]^2)` instead of the true matrix inverse. Silently corrupts the `precision_` field of every covariance estimator in the crate and all `mahalanobis(...)` distances when features are correlated. Five conformance tests `#[ignore]`d pending fix.
- *(filed, investigation)* **#337** — `MinCovDet` location/covariance diverges from sklearn FastMCD by more than expected subset-selection variance. Needs triage to determine whether bug or acceptable divergence.

## [0.3.0] - 2026-04-29

Workspace-wide parity audit against scikit-learn 1.8.0, accompanied by a 4×
expansion of the Python bindings (12 → 54 estimators) and a new dual-library
benchmark harness that runs ferrolearn and scikit-learn head-to-head in one
process across 144 paired measurements.

### Added
- **ferrolearn-bench**: Head-to-head benchmark harness — `head_to_head_full.py` runs all 54 bound estimators against their scikit-learn equivalents in a single Python process with identical datasets, hyperparameters, train/test splits, and quality metrics. Companion `render_head_to_head.py` produces Markdown reports. Per-bench JSON snapshots preserved under `ferrolearn-bench/reports/`. (#330)
- **ferrolearn-python**: 42 new PyO3 bindings — Python now exposes 54 sklearn-compatible estimators (was 12). New: `ARDRegression`, `BayesianRidge`, `HuberRegressor`, `QuantileRegressor`, `RidgeClassifier`, `LinearSVC`, `QuadraticDiscriminantAnalysis`, `MultinomialNB`, `BernoulliNB`, `ComplementNB`, `DecisionTreeRegressor`, `ExtraTreeClassifier`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `RandomForestRegressor`, `AdaBoostClassifier`, `BaggingClassifier`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `KNeighborsRegressor`, `NearestCentroid`, `MiniBatchKMeans`, `DBSCAN`, `AgglomerativeClustering`, `Birch`, `GaussianMixture`, `IncrementalPCA`, `TruncatedSVD`, `FastICA`, `NMF`, `KernelPCA`, `SparsePCA`, `FactorAnalysis`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `PowerTransformer`, `KernelRidge`, `Nystroem`, `RBFSampler`. (#330)
- **BENCHMARKS.md**: Comprehensive head-to-head report across all 144 paired measurements, with per-family geomean speedups and per-row timings + accuracy/R²/ARI deltas. (#331)

### Changed (sklearn parity fixes — measured before/after)
- **ferrolearn-preprocess**: `QuantileTransformer` forward-transform now value-matches sklearn `_data.py` (three fixes). (1) Normal output replaces the Abramowitz-Stegun `probit` (~1e-4 error) with Acklam's inverse-normal-CDF (~1e-9) + clip to ±5.199337582605575 (= `norm.ppf(1e-7-spacing)`), matching `stats.norm.ppf` (`:2855-2862`) — `[1,2,3,4,5]` Normal now `[-5.199337582605575,-0.6744897501960817,0,0.6744897501960817,5.199337582605575]` (#1320). (2) `interpolate_cdf` averages forward + reversed `np.interp` so a plateau maps to its midpoint, matching `_transform_col` (`:2843-2846`) — tied `[1,2,2,2,3]` `transform(2.0)` now `0.5` (was `0.75`) (#1321). (3) `fit` landmarks replicate numpy `np.linspace` references (`i*step`+endpoint pin) and `np.nanpercentile(X, references_*100)`'s `*100/100` round-trip (`:2694,:2702,:2795`) — fixes a ~0.083 transform divergence on tied data near plateau edges (#1322). Three-round acto-critic-verified over an 84-case stress matrix (uniform exact, normal within ~2.3e-9). (closes #1320, #1321, #1322)
- **ferrolearn-preprocess**: `KNNImputer` imputation now value-matches sklearn `_knn.py` + `nan_euclidean_distances` across the implemented surface (five fixes). (1) `partial_euclidean_distance` now scales `sqrt(sum_sq · n_features / n_valid)` instead of raw `sqrt(sum_sq)`, matching `nan_euclidean_distances` (`pairwise.py:539-547`) — fixes neighbor ordering + distance weights when rows have differing present-feature counts (#1305). (2) the no-reachable-donor branch imputes the masked training column mean, not `0.0`, matching `_knn.py:329-337` (#1306). (3) `fit` no longer errors on `n_neighbors > n_samples` (sklearn clamps, `_knn.py:349`) (#1307). (4) the `Distance` exact-match (distance 0) handling gives the matched donor weight 1 and all others 0 (was a `1e12` blend that leaked far donors), matching `_get_weights` (`neighbors/_base.py:119-121`) (#1308). (5) inf-distance (no-shared-feature) potential donors are included in the uniform average to fill the `n_neighbors` quota exactly like sklearn `argpartition` (`_knn.py:184-204`) (#1309). In-module `test_knn_imputer_too_many_neighbors_error` rewritten to `test_knn_imputer_too_many_neighbors_ok` (R-HONEST-4). Three-round acto-critic-verified: 33 oracle green guards value-match within ~1e-9. Carve-out (#1310): exact-distance-tie donor selection follows numpy `argpartition`'s unspecified tie order + ULP float noise — a documented edge, not a meaningful parity target. (closes #1305, #1306, #1307, #1308, #1309)
- **ferrolearn-preprocess**: `RFE::new` no longer errors when `n_features_to_select > n_features` — it now clamps to `n_features` and keeps ALL features, matching sklearn's warn-and-keep-all behavior (`_rfe.py:290-297`; the `while sum(support) > n_features_to_select` loop at `:314` never runs). Before: `RFE::new(&[0.5,0.3], 5, 1)` → `Err(InvalidParameter)`; after: → `Ok` with `support()==[true,true]`, `ranking()==[1,1]` (all kept). The `n_features_to_select == 0` guard is preserved. In-module `test_rfe_n_features_too_large_error` rewritten to `test_rfe_n_features_too_large_keeps_all` (R-HONEST-4). The UserWarning has no Rust analog (no log facade). Two-round acto-critic-verified CLEAN (clamp boundary stress: ==n / ==n+1 / 100-of-4 keep-all, valid count unperturbed, zero still errors). (closes #1296)
- **ferrolearn-preprocess**: `SequentialFeatureSelector::fit` validation now matches sklearn's `_sequential.py` contract. (1) The `n_features_to_select` count guard rejects `>= n_features` (was `> n_features`), mirroring `must be < n_features` (`:227-228`) — `n_features_to_select == n_features` now errors (was: selected all features). (2) Added an `ensure_min_features=2` guard (`:214`): a 1-feature X now errors with "minimum of 2 is required", placed before the count guard to match sklearn's validation precedence. In-module `test_select_all_features` rewritten to `test_select_all_features_rejected` (R-HONEST-4). Two-round acto-critic-verified CLEAN (incl. the precedence probe: a 1-feature X with an oversized count yields the min-features error first, not the count error). (closes #1284, #1285)
- **ferrolearn-preprocess**: `SelectPercentile::fit` now selects features via sklearn's `_get_support_mask` threshold rule (`_univariate_selection.py:669-686`) instead of a `ceil` rank-top-k. It computes `threshold = np.percentile(scores, 100 - percentile)` (new local `numpy_percentile` linear-interpolation helper), keeps `scores > threshold` (strict), and fills threshold-equal ties in ascending index order up to `int(n*percentile/100)`, with `percentile==100→all`/`percentile==0→none` short-circuits. Before: 5-feature fixture at `percentile=50` selected `[0,1,3]` (`k=ceil(2.5)=3`); after: `[0,3]` (matching `scores > median`). In-module `test_select_percentile_selects_highest_scoring` re-grounded against a finite-score live oracle (R-HONEST-4). Two-round acto-critic-verified bit/value-CLEAN across 8-feature non-round percentiles, exact tie-fill, both interpolation branches, and the f32 path. (closes #1274)
- **ferrolearn-preprocess**: `TargetEncoder::fit` now computes `global_mean` with a NumPy-faithful pairwise summation (matching `np.mean`/`np.add.reduce`) instead of a naive sequential left-fold, so the global mean bit-matches sklearn on mixed-magnitude targets. Before: `mean([1e16, 1.0×100])` → `99009900990099.02`; after → `99009900990099.84` (= `float(np.mean(...))`, ~12 ULP closer). New `pairwise_sum` helper mirrors numpy `pairwise_sum` (n<8 sequential, 8≤n≤128 8-way unrolled, n>128 recursive split). Two-round acto-critic-verified bit-exact across all size branches (n<8/=8/13/101/=128/300/1000). (closes #1261)
- **ferrolearn-preprocess**: `TargetEncoder::fit` per-category encoding now seeds the accumulator with `smooth*global_mean` and divides by `smooth+count` — `(smooth*y_mean + Σyᵢ)/(smooth+count)` — matching sklearn `_target_encoder_fast.pyx:60-75` exactly, instead of the algebraically-rearranged `(count*(sum/count) + smooth*global_mean)/(count+smooth)` which lost up to 1 ULP on mixed-magnitude category targets. Before/after on the critic's mixed-magnitude fixture: ferro `0x426e6ee0f70a59c9` → `0x426e6ee0f70a59ca` == sklearn `encodings_[0][0]=1045674047570.8059`. Two-round acto-critic-verified bit-exact on the f64 path. (closes #1262)
- **ferrolearn-preprocess**: `FittedRobustScaler::transform` now maps a zero-IQR column to a centered value (effective scale 1) instead of leaving it unchanged — uses `scale_eff = if iqr==0 {1} else {iqr}` and always centers (`(x-median)/scale_eff`), matching sklearn `_handle_zeros_in_scale` (scale 0→1) + center-first (`_data.py:88,1635,1673-1675`). Before: `fit_transform([[7,1],[7,2],[7,3]])` col 0 stayed 7.0; after: → 0.0. A non-constant zero-IQR column `[[1],[1],[1],[1],[9]]` now → `[[0],[0],[0],[0],[8]]` (was unchanged). R-HONEST-4: in-module `test_zero_iqr_column_unchanged` replaced by `test_zero_iqr_column_centered_to_zero`. Reflects through the `_RsRobustScaler` PyO3 binding; same pattern as StandardScaler #1191 / MinMaxScaler #1170; two-round acto-critic-verified CLEAN. (closes #1248)
- **ferrolearn-preprocess**: `FittedLabelBinarizer::transform` now silently ignores labels not seen during fit (leaving an all-zero row) instead of returning `InvalidParameter`, matching sklearn `label_binarize` `np.isin` (`_label.py:556-559`). Before: `fit([0,1,2]).transform([0,3])` → Err; after: → `Ok([[1,0,0],[0,0,0]])`. In-module `test_transform_unknown_label_error` rewritten to `test_transform_unknown_label_ignored` (R-HONEST-4). (closes #1239)
- **ferrolearn-preprocess**: `FittedLabelBinarizer::transform` single-class (k==1) now returns an all-zero `(n,1)` column (neg_label) instead of an all-`1.0` column, matching sklearn `label_binarize` n_classes==1 case (`np.zeros((n,1)); Y += neg_label`, `_label.py:532-538`). Before: `fit_transform([5,5,5])` → `[[1],[1],[1]]`; after: → `[[0],[0],[0]]`. In-module `test_single_class` tightened to assert values (R-HONEST-4). (closes #1240)
- **ferrolearn-preprocess**: `FittedLabelBinarizer::inverse_transform` binary branch now uses a strict `> 0.5` threshold instead of `>= 0.5`, matching sklearn `_inverse_binarize_thresholding` `y > threshold` (default 0.5) (`_label.py:667`). Before: `inverse_transform([[0.5]])` for `fit([0,1])` → class 1; after: → class 0 (`0.5 > 0.5` is False). (closes #1241)
- **ferrolearn-preprocess**: `FittedMultiLabelBinarizer::transform` now silently ignores labels not seen during fit instead of returning `FerroError::InvalidParameter`, matching sklearn's `_transform` collect-and-ignore (`_label.py:889-902`). Before: `fit([[0,1]]).transform([[0,5]])` → Err; after: → `Ok([[1,0]])` (label 5 skipped). The Python `warnings.warn("unknown class(es) ... will be ignored")` has no Rust analog (crate has no log facade) and is intentionally not emitted. In-module `test_transform_unknown_label_error` rewritten to `test_transform_unknown_label_ignored` (R-HONEST-4). (closes #1230)
- **ferrolearn-preprocess**: `FittedMultiLabelBinarizer::inverse_transform` now validates the indicator matrix is all 0s/1s (returns `InvalidParameter` on any other value) and includes a class iff its cell `== 1.0`, matching sklearn `np.setdiff1d(yt,[0,1])` raise + exact-1 selection (`_label.py:941-947`). Before: `inverse_transform([[0.4,0.6,0.5]])` → `Ok([[1,2]])` via a `>= 0.5` threshold; after: → Err (sklearn raises ValueError). In-module `test_inverse_threshold` rewritten to `test_inverse_rejects_non_01` (R-HONEST-4). (closes #1231)
- **ferrolearn-preprocess**: `CountVectorizer::tokenize` now mirrors sklearn's default `token_pattern=r"(?u)\b\w\w+\b"` (text.py:1161) — drops length-1 tokens and treats `_` as a word character (`split(|c| !(c.is_alphanumeric() || c == '_')).filter(|s| s.chars().count() >= 2)`), instead of splitting on every non-alphanumeric and keeping single-char tokens. Before: `['foo a bar']` → `['a','bar','foo']`; after: `['bar','foo']` (matches live sklearn). `['a_b cd']` before → `['a','b','cd']`; after → `['a_b','cd']`. In-module `test_count_vectorizer_max_features` rewritten against a live oracle (R-HONEST-4). (closes #1217)
- **ferrolearn-preprocess**: `CountVectorizer` `max_df` float-proportion threshold no longer rounds — filters `(count as f64) <= max_df * n_docs` (matching sklearn `max_doc_count = max_df * n_doc`, text.py:1379) instead of `count <= ceil(max_df * n_docs)`. Before: `max_df=0.5` on 3 docs kept terms in 2 docs (`2 <= ceil(1.5)=2`); after: excludes them (`2.0 <= 1.5` false), matching live sklearn. (closes #1218)
- **ferrolearn-tree**: `RandomForestClassifier` and `RandomForestRegressor` now sample features **per-split** (Breiman 2001 / sklearn behaviour) rather than picking a single fixed feature subset per tree. Closed a -16.05pp accuracy gap at medium_10Kx100. New helper `build_classification_tree_per_split_features` / `build_regression_tree_per_split_features`. (#330)
- **ferrolearn-linear**: `LinearSVC` rewritten with **coordinate-Newton** updates replacing fixed-step (LR=0.01) gradient descent — closed a -21.05pp accuracy gap at medium_10Kx100 while running 2× faster on fit. (#330)
- **ferrolearn-kernel**: `KernelRidge` default kernel changed from `Rbf` to `Linear` to match scikit-learn's `KernelRidge(kernel='linear')` default. Closed a -0.20 R² gap at tiny scale (now exact parity). (#330)
- **ferrolearn-tree**: `AdaBoostClassifier` default algorithm changed from `SAMME.R` to `SAMME` to match scikit-learn ≥ 1.4 (which removed `SAMME.R` in 1.6). Closed a -19.00pp accuracy gap at small scale. (#330)
- **ferrolearn-cluster**: `GaussianMixture` initialisation upgraded from random-row sampling to **Greedy KMeans++** (Arthur & Vassilvitskii 2007 with `2 + log(k)` trial selection, matching sklearn's `_kmeans_plusplus`). M-step now adds `reg_covar = 1e-6` to component covariance diagonals. Closed -0.27 ARI / -0.17 / -0.16 gaps at tiny / small / medium scales (now all exact parity). (#330)
- **ferrolearn-cluster**: `MiniBatchKMeans` defaults switched to scikit-learn 1.4+ values: `batch_size 100 → 1024`, `max_iter 300 → 100`, `tol 1e-4 → 0.0`. Closed a -0.16 ARI gap at medium_5Kx20 (now exact parity). (#330)
- **ferrolearn-cluster**: `KMeans`, `MiniBatchKMeans` initialisations upgraded to **Greedy KMeans++** for robustness at scale. (#330)
- **ferrolearn-linear**: `QuantileRegressor` IRLS L1 penalty now scaled by `n_samples` so the user-facing `alpha` parameter is numerically equivalent to scikit-learn's. Previously `alpha=1.0` in ferrolearn was effectively `~1/n` of sklearn's `alpha=1.0`. (#332)

### Workspace
- All workspace crates bumped from 0.2.2 → 0.3.0. (#329)
- Workspace test count: **3,662 tests passing**, 0 failing.

### Bench results — geomean speedups vs scikit-learn 1.8.0 (n=144 paired runs)

| Family | n | fit geomean | predict geomean | mean Δ score |
|---|---:|---:|---:|---:|
| regressor | 43 | **8.21×** | **4.39×** | -0.0006 R² |
| classifier | 51 | **6.75×** | **8.88×** | +0.0035 accuracy |
| cluster | 15 | 1.35× | — | +0.0000 ARI (exact parity, 15/15) |
| decomp | 15 | **5.16×** | **4.56×** | — |
| preprocess | 14 | **9.82×** | **2.74×** | — (numerical agreement to 1e-16) |
| kernel approx | 6 | **6.78×** | 1.26× | — |

## [0.2.2] - 2026-04-29

Coordinated workspace bump for all crates from `0.2.0` (and `ferrolearn-bayes 0.2.1`) to `0.2.2`. Includes the conjugate-priors module previously released as `ferrolearn-bayes 0.2.1`, GP-classifier feature completion, and a workspace-wide maintenance pass.

### Added
- **ferrolearn-kernel**: `GaussianProcessClassifier::log_marginal_likelihood()` — Laplace-approximation log marginal likelihood (Rasmussen & Williams eq. 3.32 / Algorithm 5.1), summed across one-vs-rest binary models for multiclass. Standard objective for kernel hyperparameter selection and model comparison (#237)
- **ferrolearn-kernel**: `FittedGaussianProcessClassifier::classes()` accessor returning sorted class labels (#237)
- **ferrolearn-kernel**: Expose `KernelRidge`/`FittedKernelRidge` (dual-form kernel ridge regression with RBF/Polynomial/Linear/Sigmoid/Laplacian/Cosine kernels), `Nystroem`/`FittedNystroem`/`KernelType` (low-rank Nyström kernel approximation), and `RBFSampler`/`FittedRBFSampler` (random Fourier features per Rahimi & Recht 2007) — these implementations were already in the source tree but the parent modules were not declared in `lib.rs`, so they were unreachable from outside the crate. Activates 52 previously-dormant tests (#4)
- **ferrolearn-bayes**: Conjugate priors module with closed-form posterior updates (`ferrolearn_bayes::conjugate`) (#235, originally released as ferrolearn-bayes 0.2.1)
  - `posterior_normal_normal` — Normal-Normal conjugate update for the latent mean of a Gaussian likelihood with known per-observation variance, given a Normal prior on the mean.
  - `NormalNormalPosterior { mean, var }` — typed posterior summary.

### Changed
- preprocess/standard_scaler: StandardScaler::transform now maps a constant (zero-variance) column to 0 instead of leaving it unchanged — uses effective scale 1 (s_eff = if s==0 {1} else {s}) so (x-mean)/1 = 0, matching sklearn _handle_zeros_in_scale (scale_=1 on constant cols) + with_mean centering (_data.py:88,1019-1021,1064-1067). inverse_transform aligned to x*s_eff+mean (matches sklearn X *= scale_, so inverse of non-round-trip input on a constant col gives x+mean not a collapse). Was leaving constant columns at their original value (fit_transform([[1,5],[2,5],[3,5]]) col 1 stayed 5.0; sklearn → 0.0). R-HONEST-4: in-module test_zero_variance_column_unchanged replaced by test_constant_column_maps_to_zero. Two-round acto-critic-verified CLEAN; reflected through _RsStandardScaler PyO3 binding (maturin smoke test) (#1191, #1190)
- preprocess/polynomial_features: PolynomialFeatures::transform now validates input like sklearn's check_array — rejects (in sklearn order) zero samples → FerroError::InsufficientSamples, zero features → InvalidParameter, and non-finite NaN/±inf → InvalidParameter, matching transform → _validate_data (_polynomial.py:433-435). Was returning Ok with NaN/inf rows and accepting zero-row arrays where sklearn raises ValueError. Input-only validation: a finite input whose polynomial product overflows to inf is correctly accepted (matches sklearn). Mirrors converged binarizer/normalizer; two-round acto-critic-verified CLEAN (#1180, #1179)
- preprocess/min_max_scaler: MinMaxScaler::transform now maps a constant (zero-range) column to feature_range[0] instead of leaving it unchanged, matching sklearn's _handle_zeros_in_scale (data_range 0→1, _data.py:88,508-511; transform(data_min)=fr[0]). Was leaving constant columns at their original value (e.g. fit_transform([[5,1],[5,2],[5,3]]) col 0 stayed 5.0; sklearn → 0.0). R-HONEST-4: in-module test_zero_range_column_unchanged (pinned the wrong 5.0) replaced by test_constant_column_maps_to_range_min. Two-round acto-critic-verified CLEAN (default/(−1,1)/(2,5)/negative/single-row-fit); reflected through the _RsMinMaxScaler PyO3 binding (maturin smoke test) (#1170, #1169)
- preprocess/normalizer: Normalizer::transform now validates input like sklearn's check_array — rejects (in sklearn order) zero samples → FerroError::InsufficientSamples (validation.py:1084), zero features → InvalidParameter (validation.py:1093), and non-finite NaN/±inf → InvalidParameter (validation.py:1063 force_all_finite), matching Normalizer.transform → normalize → check_array (_data.py:1933-1940). Was returning Ok with NaN/inf rows and accepting empty/zero-feature arrays where sklearn raises ValueError. Finite/zero-norm-row inputs not over-rejected; mirrors converged binarizer.rs; two-round acto-critic-verified CLEAN; the PipelineTransformer consumer inherits the validation (#1140, #1139)
- preprocess/label_encoder: LabelEncoder::fit no longer rejects empty input — removed the `if x.is_empty()` → InsufficientSamples guard so fit([]) yields an empty FittedLabelEncoder (classes=[], n_classes=0), matching sklearn LabelEncoder().fit([]) (classes_=[], no error; _label.py:98 _unique([])). R-HONEST-4: the in-module test_empty_input_error (which pinned the divergent rejection) was replaced by test_empty_fit_yields_empty_classes. Two-round acto-critic-verified CLEAN incl. post-empty-fit transform/inverse behavior (#1134, #1133)
- preprocess/binarizer: Binarizer::transform now validates input like sklearn's check_array — rejects (in sklearn's order) zero samples → FerroError::InsufficientSamples (validation.py:1084 min-samples, #1124), zero features → InvalidParameter (validation.py:1093 min-features, #1125), and non-finite NaN/±inf → InvalidParameter (validation.py:1063 force_all_finite, #1123), matching Binarizer.transform's _validate_data (_data.py:2301). Was silently binarizing non-finite input (NaN→0, +inf→1) and accepting empty/zero-feature arrays where sklearn raises ValueError. Finite extremes (1e308/-0.0/subnormal) not over-rejected; two-round acto-critic-verified CLEAN (#1122)
- preprocess: cleared 2 pre-existing collapsible_if clippy lints (CI-breaking on stable 1.95) blocking the ferrolearn-preprocess crate gauntlet — count_vectorizer.rs (max_features prune guard) and tfidf.rs (idf shape-check guard) nested `if let X { if Y }` collapsed to let-chains `if let X && Y` (MSRV 1.88, behavior-preserving, acto-critic-verified semantically identical) (#1111)
- linear/lib: RegressorScore::score (r2_score) constant-y_true edge now returns 0.0 (was neg_infinity) when ss_tot==0 ∧ ss_res!=0, matching sklearn.metrics.r2_score (sklearn/metrics/_regression.py:891, the RegressorMixin.score delegate base.py:849); zero-residual stays 1.0. Live oracle r2_score([5,5,5],[4,5,6])=0.0 (#1104)
- linear/lib: log_proba (body of every classifier predict_log_proba) is now the unclamped p.ln(), matching sklearn predict_log_proba = np.log(predict_proba) (sklearn/discriminant_analysis.py:1059) — was clamping p≤1e-300 to ln(1e-300)≈-690.78; a 0.0 probability now maps to -inf (observable on QDA zero-proba). Also drops an incidental F::from(1e-300).unwrap() (#1105)
- cluster/gmm: GaussianMixture Full/Tied Gaussian log-density no longer double-counts log|Σ| — the Full|Tied arm of fn log_responsibilities now adds log_w + log_norm - 0.5·maha (log_norm already folds in -0.5·log_det, matching sklearn _estimate_log_gaussian_prob, sklearn/mixture/_gaussian_mixture.py:448-507), was subtracting 0.5·log_det TWICE; and fn cholesky no longer re-adds reg=1e-6 on top of the M-step reg_covar (the density path was regularizing Σ twice). score/score_samples/lower_bound_/aic/bic now value-match the oracle (2-blob fixture: score 1.8696902967180025 == sklearn, was 7.269941). Diag/Spherical density arms were already correct (#1093)
- cluster/optics: OPTICS traversal now uses sklearn's single-pool linear-argmin (smallest-index tie-break, no heap, sklearn/cluster/_optics.py:638-659) + np.around(decimals=precision) round-ties-even reachability rounding (:711) — was a BinaryHeap + component-restart loop that diverged from sklearn's ordering_/reachability_ on reachability ties. ordering_ and reachability_ now value-match the oracle (#1080)
- cluster/hdbscan: HDBSCAN core distance now uses sorted_dists[min_samples-1] (the min_samples-NN query includes self at index 0 = sklearn neighbors_distances[:, -1], sklearn/cluster/_hdbscan/hdbscan.py:351-352) — was dists[min_samples] (one neighbor too far, so ferrolearn min_samples=k reproduced sklearn k+1) (#1070)
- cluster/mini_batch_kmeans: MiniBatchKMeans default n_init is now 1 (= sklearn n_init="auto" → 1 for the default init="k-means++", sklearn/cluster/_kmeans.py:886-888) — was 3 (a mis-translation; the code comment is corrected) (#1047)
- cluster/kmeans: KMeans.fit now runs a final E-step (re-assigns labels_/inertia_ to the converged centers) after the Lloyd loop so labels_/inertia_ are consistent with cluster_centers_ and fit(X).predict(X) == labels_ — mirroring sklearn's post-loop E-step re-run (sklearn/cluster/_kmeans.py:605-625); previously cluster_centers_ was one M-step ahead of labels_ (#1037)
- cluster/kmeans: KMeans default n_init is now 1 (= sklearn n_init="auto" → 1 for the default init="k-means++", sklearn/cluster/_kmeans.py:886-896) — was 10 (#1045)
- cluster/bisecting_kmeans: BisectingKMeans default bisecting_strategy is now LargestSSE (= sklearn "biggest_inertia", sklearn/cluster/_bisect_k_means.py:229) — was LargestCluster (sklearn's non-default) (#1025)
- cluster/bisecting_kmeans: BisectingKMeans default n_init is now 1 (= sklearn n_init=1, sklearn/cluster/_bisect_k_means.py:222) — was 10 (#1026)
- cluster/label_spreading: LabelSpreading now rejects alpha=0 (and alpha=1) — the open interval (0,1) matching sklearn _parameter_constraints alpha=Interval(Real,0,1,closed="neither") (sklearn/semi_supervised/_label_propagation.py:585); was accepting alpha=0. test_alpha_zero_recovers_initial rewritten to test_alpha_zero_rejected (R-HONEST-4) (#1009)
- cluster/label_spreading: LabelSpreading now derives classes_ = sorted unique non-(-1) labels, n_classes = len(classes_), one-hot indexed by class position, and maps the final argmax through classes_ (= sklearn classes_/transduction_, sklearn/semi_supervised/_label_propagation.py:272-274,333) — was n_classes=max(label)+1 with raw-argmax-index labels, reporting n_classes=3 with a phantom class on non-contiguous label sets (e.g. {0,2}). Now n_classes()==2 and labels ⊆ {0,2} matching sklearn (#1011)
- cluster/label_propagation: LabelPropagation now derives classes_ = sorted unique non-(-1) labels, n_classes = len(classes_), one-hot indexed by class position, and maps the final argmax index through classes_ (= sklearn classes_/transduction_, sklearn/semi_supervised/_label_propagation.py:272-274,333) — was n_classes=max(label)+1 with raw-argmax-index labels, which on non-contiguous label sets (e.g. {0,2}) reported n_classes=3 with a phantom class and could emit labels not in the input set. Now n_classes()==2 and labels ⊆ {0,2} matching sklearn (#999)
- cluster/mean_shift: MeanShift bandwidth=None now estimates bandwidth via sklearn's kNN heuristic (estimate_bandwidth(X, quantile=0.3): mean over points of the max distance among each point's int(n*0.3) nearest neighbors, sklearn/cluster/_mean_shift.py:95-106) — was the median of all pairwise distances. estimate_bandwidth is now a pub fn (default-path VALUE matches sklearn to ~1e-12; n_samples/random_state subsampling deferred). On a 3-blob fixture auto-MeanShift now yields sklearn's 3 clusters, was 1 (#985)
- cluster/affinity_propagation: AffinityPropagation default preference (preference=None) now medians the FULL n×n affinity matrix (n zero self-distances + each off-diagonal twice = np.median(affinity_matrix_), sklearn/cluster/_affinity_propagation.py:519-520) — was medianing only the off-diagonal upper triangle, giving -13.0 instead of -9.0 on the docstring X and k=2 instead of sklearn's k=3 on borderline blobs (#971)
- cluster/feature_agglomeration: FeatureAgglomeration::fit now rejects <2 features (sklearn _validate_data ensure_min_features=2, sklearn/cluster/_agglomerative.py:1338) — was accepting a 1-feature X (#939 #944)
- cluster/spectral: SpectralClustering::fit now allows gamma=0 (rejects only gamma<0, matching sklearn gamma Interval(Real,0,None,closed="left"), sklearn/cluster/_spectral.py:612) — was over-rejecting gamma<=0 (#930 #936)
- bayes/categorical: CategoricalNB::fit now allows alpha=0 (rejects only alpha<0, matching sklearn Interval(Real,0,None,closed="left"), naive_bayes.py:1333) and rejects negative feature values with "Negative values in data passed to CategoricalNB (input X)" (sklearn _check_X_y check_non_negative, :1435-1440) — was silently mapping negatives to category 0 (#921 #922)
- bayes/complement: ComplementNB::fit now rejects alpha<0 with InvalidParameter — sklearn inherits alpha: Interval(Real, 0, None, closed="left") (naive_bayes.py:530, :1000-1003), a hard >=0 reject at fit; alpha=0 still allowed (#914 #918)
- bayes/bernoulli: BernoulliNB::new() now defaults binarize=0.0 (binarize at 0, sklearn/naive_bayes.py:1164) — was None; and BernoulliNB::fit rejects alpha<0 (sklearn Interval(Real,0,None,closed="left"), :530), alpha=0 still allowed (#906 #907 #911 #912)
- bayes/multinomial: MultinomialNB::fit now rejects alpha<0 with InvalidParameter — sklearn declares alpha: Interval(Real, 0, None, closed="left") (naive_bayes.py:530), a hard >=0 reject at fit; alpha=0 still allowed (#900 #904)
- bayes/gaussian: GaussianNB::fit epsilon_ corrected to var_smoothing·np.var(X,axis=0).max() (global per-feature population variance, no 1.0 floor; sklearn/naive_bayes.py:431) and priors now validated for sum≈1 + non-negativity, not just length (:448-455) (#891 #893)
- neighbors/radius_neighbors: RadiusNeighborsRegressor::predict returns NaN for no-neighbor rows instead of raising (sklearn/neighbors/_regression.py:482); RadiusNeighborsClassifier::predict_proba leaves empty-neighborhood rows all-zero (not uniform) when outlier_label is absent/out-of-class (_classification.py:813-829) (#882 #881)
- neighbors/knn: KNeighborsClassifier::fit and KNeighborsRegressor::fit no longer error when n_neighbors > n_samples — sklearn defers that check to predict/kneighbors() call time (sklearn/neighbors/_base.py:828); removed the non-sklearn fit-time InsufficientSamples guard from both fits, kept the n_neighbors>=1 + X/y shape validation (#874 #879)
- neighbors/nearest_neighbors: NearestNeighbors::fit no longer errors when n_neighbors > n_samples — sklearn defers that check to kneighbors() call time (sklearn/neighbors/_base.py:828); removed the non-sklearn fit-time InsufficientSamples guard, kept the n_neighbors>=1 fit validation (#872)
- LOF score_samples/decision_function convention (#849)
- LOF predict/fit_predict labels via negative_outlier_factor_<offset_ (#848)
- LOF offset_ ('auto'=-1.5 + percentile) (#847)
- Divergence: ferrolearn-neighbors NearestCentroid::fit shrink centroids diverge from sklearn/neighbors/_nearest_centroid.py:226-227 vs :183-184 on partial-constant features (#840)
- Divergence: ferrolearn-neighbors::NearestCentroid::fit clamps zero-variance to 1.0 instead of raising (sklearn/neighbors/_nearest_centroid.py:174-175) (#839)
- Divergence: ferrolearn-neighbors::NearestCentroid::fit accepts single-class y (sklearn/neighbors/_nearest_centroid.py:147-151 raises ValueError) (#838)
- Divergence: ferrolearn-neighbors::NearestCentroid::fit shrink_threshold omits s += median(s) (sklearn/neighbors/_nearest_centroid.py:184) (#837)
- Divergence: ferrolearn-neighbors::radius_neighbors_graph keeps self-edge vs sklearn/neighbors/_graph.py:164 default include_self=False (#824)
- Divergence: ferrolearn-neighbors::kneighbors_graph includes self (zero-diagonal missing) vs sklearn/neighbors/_graph.py:59 default include_self=False (#823)
- Divergence: ferrolearn-metrics::top_k_accuracy_score diverges from sklearn/_ranking.py:2043 on tie-breaking (#812)
- Divergence: ferrolearn-metrics::calibration_curve diverges from sklearn/calibration.py:1035 on float-edge binning (#811)
- Divergence: ferrolearn-metrics calibration_curve bins by floor(prob*n_bins); sklearn calibration.py:1035 uses searchsorted(bins[1:-1], prob) (#810)
- Divergence: ferrolearn-metrics det_curve retains the (0,0)/+inf ROC endpoint sklearn _ranking.py:362-376 drops (#809)
- Divergence: ferrolearn-metrics roc_curve has no drop_intermediate (sklearn _ranking.py:1054 default True); keeps all distinct-score thresholds (#808)
- Divergence: ferrolearn-metrics log_loss clips to EPS=1e-15, sklearn _classification.py:2951 clips to np.finfo(float64).eps=2.22e-16 (#807)
- Divergence: ferrolearn-metrics::rand_score diverges from sklearn/metrics/cluster/_supervised.py:337 for single-sample input (#800)
- Divergence: ferrolearn-metrics homogeneity/completeness/v_measure/hcv error on empty input (sklearn returns 1.0/(1,1,1)) vs sklearn/metrics/cluster/_supervised.py:531-532 (#799)
- Divergence: ferrolearn-metrics::calinski_harabasz_score returns inf (not 1.0) when intra_disp==0 vs sklearn/metrics/cluster/_unsupervised.py:387-389 (#798)
- Divergence: ferrolearn-metrics::rand_score panics (u64 subtract overflow) vs sklearn/metrics/cluster/_supervised.py:337-343 (#797)
- Divergence: scorer.rs REQ-5 mis-frames d2_absolute_error_score as blocked on heterogeneous Scorer type #781 — it is a regression fn(&Array1,&Array1) metric registerable now (sklearn _sign==+1, _scorer.py:788) (#787)
- Divergence: ferrolearn-metrics scorer registry diverges from sklearn/metrics/_scorer.py:761 (neg_max_error is not a sklearn name; max_error missing) (#780)
- Divergence: ferrolearn-metrics::Scorer::score diverges from sklearn/metrics/_scorer.py:376 (sign not applied) (#779)
- Divergence: d2_score_with / d2_tweedie_score diverge from sklearn/metrics/_regression.py:1736-1739,1599 on n>=2 constant-y_true (zero denominator) (#771)
- Divergence: ferrolearn_metrics d2_* (absolute_error/pinball/tweedie) diverge from sklearn/metrics/_regression.py:1699-1702,1584-1587 (<2 samples -> nan, not Ok(0.0)) (#770)
- Divergence: ferrolearn_metrics::mean_absolute_percentage_error diverges from sklearn/metrics/_regression.py:403-404 (eps clamp vs skip-zero-y_true) (#769)
- Divergence: ferrolearn_metrics::explained_variance_score diverges from sklearn/metrics/_regression.py:889-891 (force_finite on constant y_true) (#768)
- Divergence: ferrolearn_metrics::r2_score diverges from sklearn/metrics/_regression.py:889-891 (force_finite on constant y_true) (#767)
- Divergence: ferrolearn-metrics::label_ranking_loss diverges from sklearn/metrics/_ranking.py:1463-1465 (degenerate-row denominator: divides by non-degenerate count instead of n_samples) (#756)
- Divergence: ferrolearn-metrics::ndcg_score diverges from sklearn/metrics/_ranking.py:1749,1868 (no tie-averaging + missing negative-y_true ValueError guard) (#755)
- Divergence: ferrolearn-metrics::dcg_score diverges from sklearn/metrics/_ranking.py:1528 (no tie-averaging; default ignore_ties=False) (#754)
- Audit: corrected hist_gradient_boosting REQ-7 overclaim — HGBC multiclass predict_proba is a float32 grad/hessian saturation artifact, NOT-STARTED (#758); fixed REQ-table cites + dropped/omitted REQ rows across tree-ensemble modules (#757)
- Divergence: ferrolearn-tree::compute_bin_edges diverges from sklearn/ensemble/_hist_gradient_boosting/binning.py:53-55 (distinct-value midpoints vs quantile interpolation) (#746)
- Divergence: gradient_boosting::huber_leaf_value median uses np.median tie (mean of two middles) instead of _weighted_percentile lower-percentile; Huber predict off by ~6.6e-4 vs sklearn (#738)
- Divergence: gradient_boosting::lad_leaf_value diverges from sklearn AbsoluteError leaf-update (uses np.median not _weighted_percentile lower-percentile) (#737)
- Divergence: ferrolearn-tree GradientBoostingRegressor(loss=Huber) diverges from sklearn/_loss/loss.py:694-710 — missing median+clipped-mean terminal-region update (#736)
- Divergence: ferrolearn-tree GradientBoostingClassifier diverges from sklearn/ensemble/_gb.py:191-206 — missing LogLoss Newton terminal-region update (#735)
- Divergence: ferrolearn-tree GradientBoostingRegressor(loss=Lad) diverges from sklearn/ensemble/_gb.py:241-247 — missing weighted-median terminal-region update (#734)
- Divergence: ferrolearn-tree FittedIsolationForest::score_samples returns NaN for single-sample fit (max_samples_==1, denominator==0) where sklearn returns -0.5 (#732)
- Divergence: ferrolearn-tree average_path_length c(2) gap vs sklearn/ensemble/_iforest.py:562 (sklearn special-cases n==2 -> 1.0; ferrolearn computes 0.1544 from general formula; in-src test_average_path_length_values asserts the WRONG 0<c(2)<1) (#727)
- Divergence: ferrolearn-tree IsolationForest::score_samples sign inversion vs sklearn/ensemble/_iforest.py:451 (returns +2^(-mean/c) in (0,1]; sklearn returns -2^(-mean/c) in [-1,0]) (#726)
- Divergence: ferrolearn-tree bagging.rs fit panics (index OOB) for max_features<1.0 — aggregate_tree_importances double-maps original feature index through feature_indices[t] (#719)
- Divergence: ferrolearn-tree FittedBaggingClassifier::predict hard-votes; sklearn BaggingClassifier.predict soft-votes (_bagging.py:913-914) (#718)
- Divergence: ferrolearn-tree adaboost::fit_samme missing perfect-fit estimator_weight=1.0 guard (_weight_boosting.py:679-680) (#710)
- Divergence: ferrolearn-tree adaboost::fit_samme fits resampled UNWEIGHTED stump vs sklearn weighted fit (_weight_boosting.py:664) (#709)
- Divergence: ferrolearn-tree AdaBoostRegressor::fit reweight exponent diverges from sklearn/ensemble/_weight_boosting.py:1209-1211 (missing * learning_rate) (#703)
- Divergence: ferrolearn-tree FittedVotingClassifier::predict tie-break diverges from sklearn/ensemble/_voting.py:445 (Rust max_by_key last-index vs numpy argmax(bincount) lowest-index) (#694)
- Divergence: ferrolearn-tree RandomTreesEmbedding::new sets n_estimators=10, sklearn default is 100 (sklearn/ensemble/_forest.py:2820) (#687)
- Divergence: ferrolearn-tree::FittedExtraTreesClassifier::predict hard-votes; sklearn ExtraTreesClassifier.predict soft-votes (_forest.py:907) (#679)
- Divergence: ferrolearn-tree RandomForestClassifier::predict hard-votes; sklearn/ensemble/_forest.py:904-907 soft-votes (argmax of mean predict_proba) (#670)
- Divergence: ferrolearn-tree ExtraTreeRegressor ignores criterion (hard-wired MSE/mean leaves) vs sklearn/tree/_criterion.pyx MAE.node_value median (#681) (#667)
- decision_tree: class_weight (None/balanced/dict) (#665)
- decision_tree: max_leaf_nodes best-first growth (#664)
- decision_tree: ccp_alpha minimal cost-complexity pruning (#663)
- decision_tree: min_impurity_decrease + min_weight_fraction_leaf stopping gates (#662)
- decision_tree: REQ-1 alt criteria (log_loss/friedman_mse/absolute_error/poisson) (#661)
- Divergence: ferrolearn-tree decision_tree split tie-break ignores random_state feature-order (sklearn/tree/_splitter.pyx:293) (#659)
- Divergence: ferrolearn-tree decision_tree missing FEATURE_THRESHOLD=1e-7 constant-feature band (sklearn/tree/_splitter.pyx:33) (#660)
- Divergence: FittedNuSVC/FittedNuSVR do not re-expose support()/dual_coef()/intercept()/n_support() libsvm-layout fitted attrs (#657)
- Divergence: NuSVR missing C parameter (sklearn/svm/_classes.py:1531 default C=1.0); ferrolearn forces C=1/(nu*n) (#656)
- Divergence: NuSVR::predict diverges from sklearn/svm/src/libsvm/svm.cpp solve_nu_svr (nu_svm.rs delegates to epsilon-SVR) (#655)
- Divergence: NuSVC::decision_function diverges from sklearn/svm/src/libsvm/svm.cpp solve_nu_svc (nu_svm.rs delegates to SVC C=1/(nu*n)) (#654)
- translate(one_class_svm): REQ-6 constructor params/defaults (max_iter -1, cache_size 200, gamma/shrinking) (#651)
- translate(one_class_svm): REQ-5 nu param + (0,1] validation pin (#650)
- translate(one_class_svm): REQ-4 decision_function + score_samples + predict sign (#649)
- translate(one_class_svm): REQ-3 fitted attrs support_/support_vectors_/n_support_/dual_coef_/intercept_/offset_/coef_ (#648)
- translate(one_class_svm): REQ-1 dual_coef_/rho scaling — normalized (sum a=1) vs libsvm un-normalized (sum a=nu*n) (#646)
- translate(one_class_svm): REQ-2 gamma='scale'/'auto' not resolved at fit (uses 1.0) (#647)
- translate(svm): REQ-9 probability Platt scaling predict_proba (#642)
- translate(svm): REQ-8 estimator-level param surface + defaults (R-DEV-2) (#641)
- translate(svm): REQ-5 predict ovo voting tie-break (lower class index) (#638)
- translate(svm): REQ-4 decision_function shape/sign + ovr transform (#637)
- translate(svm): REQ-7 multiclass one-vs-one per-pair coef pin (#640)
- translate(svm): REQ-6 epsilon-SVR fitted attrs + oracle pin (#639)
- translate(svm): REQ-3 expose libsvm-layout fitted attrs + binary sign flip (#636)
- translate(svm): REQ-2 pin C-SVC SMO fit (dual_coef_/intercept_/support_) vs live oracle (#635)
- translate(svm): REQ-1 gamma scale/auto resolution at fit time (kernels resolve None->1.0) (#634)
- chore(clippy): cleared all Rust 1.95 lints blocking `ferrolearn-linear` crate `-D warnings` (omp.rs/svm.rs collapsible_if, linalg.rs assign_op, lda.rs test needless_range_loop, test-fixture literal reformat) — behavior-preserving (#378, #357)
- Translation unit: ferrolearn-linear/src/isotonic.rs — out_of_bounds='nan' default + _make_unique weighted collapse (#573)
- Divergence: SGDRegressor/SGDClassifier do not validate l1_ratio to [0,1] (sklearn/linear_model/_stochastic_gradient.py:2018,1217) (#540)
- Divergence: SGD Hinge::gradient diverges from sklearn/linear_model/_sgd_fast.pyx.tp:224 at z==threshold boundary (#539)
- translate: ferrolearn-linear/ransac.rs — RANSACRegressor sklearn parity (iter 24) (#511)
- QuantileRegressor: scale alpha by n_samples for sklearn parity (#332)
- Blocker for REQ-1/REQ-3 of quantile_regressor: intercept recovered via X/y centering is invalid for quantile regression (sklearn _quantile.py:177 'centering y and X does not work for quantile regression'). ferrolearn's FittedQuantileRegressor intercept is computed as y_mean - x_mean.dot(w), giving the SAME intercept for every quantile; sklearn's LP makes the intercept a free LP variable (s0-t0). Live oracle q=0.8 alpha=0: ferro intercept=0.2988 vs sklearn 0.8815 (3x). Fix: fit intercept as an LP variable, not by centering — requires the LP solver (#340). (#506)
- Blocker for REQ-5 of huber_regressor: outliers_ mask (|resid| > scale*epsilon) not computed/exposed (#497)
- Blocker for REQ-4 of huber_regressor: scale_ not jointly optimized/bounded — IRLS has no sigma parameter (#496)
- Blocker for REQ-1 of huber_regressor: ferrolearn IRLS diverges from sklearn L-BFGS Huber on outlier data (no joint scale optimization) (#495)
- Blocker for REQ-2 of omp: default n_nonzero_coefs must be max(int(0.1*n_features),1) when both None (sklearn _omp.py:785); ferrolearn errors instead (#488)
- Blocker for REQ-2 of omp: default n_nonzero_coefs must be max(int(0.1*n_features),1) when both None (sklearn _omp.py:785); ferrolearn errors instead (#488)
- Blocker for REQ-2 of lars: LassoLars uses forward-stepwise OLS (ols_active), not the equiangular LARS-lasso path; coef_ diverges from sklearn LassoLars on diabetes (a=0.1: feat-4 enters in ferrolearn, feat-9 in sklearn; -233 vs -155) (#482)
- Blocker for REQ-3 of ard: needs per-iteration keep_lambda pruning (lambda_>=threshold_lambda drops columns from the solve each iter, sklearn _bayes.py:691-692); ferrolearn prunes coef once after the loop (#476)
- Blocker for REQ-2 of ard: needs init alpha_=1/(np.var(y)+eps) (sklearn _bayes.py:658); ferrolearn fn fit hardcodes alpha=F::one() (#475)
- Blocker for REQ-1 of ard: needs iterative keep_lambda column-masking + init alpha=1/(var(y)+eps) + convergence on sum|coef_old-coef_|<tol to match sklearn ARDRegression.fit coef_/alpha_/lambda_ (2D parity fails: feature 0 wrongly pruned) (#474)
- Blocker for REQ-3 of bayesian_ridge: alpha_init default is 1.0 instead of sklearn's None->1/(Var(y)+eps); changes EM trajectory and fitted alpha_/lambda_/coef_ (#466)
- Blocker for REQ-2 of bayesian_ridge: BayesianRidge<F> lacks alpha_1/alpha_2/lambda_1/lambda_2 Gamma-prior params (sklearn defaults 1e-6); they enter the alpha_/lambda_ update equations (#465)
- Blocker for REQ-1 of bayesian_ridge: fit update equations omit Gamma hyperpriors (2*alpha_1/2*alpha_2/2*lambda_1/2*lambda_2) and use a trace/Cholesky-diag gamma approximation instead of sklearn's exact SVD eigenvalue formula; alpha_/lambda_/coef_ diverge from sklearn BayesianRidge (#464)
- Blocker for REQ-5 of logistic_regression_cv: stratified_kfold_split uses i%k-within-class, diverges from sklearn StratifiedKFold balanced partition (contiguous chunks per class, optional shuffle/random_state) — different fold membership changes per-C accuracy and selected C_ (#456)
- Blocker for REQ-14 of elastic_net_cv: l1_ratio=0 alpha-grid path; sklearn _alpha_grid raises ValueError for l1_ratio=0 (auto grid unsupported), ferrolearn silently uses max|Xᵀy|/n (#440)
- Blocker for REQ-5 of elastic_net_cv: kfold_indices uses round-robin i%k; sklearn KFold is contiguous blocks (#431)
- Blocker for REQ-6 of elastic_net_cv: ElasticNetCV::new() defaults to 7-element l1_ratios grid; sklearn default is l1_ratio=0.5 (single) (#432)
- Blocker for REQ-5 of lasso_cv.md: LassoCV uses round-robin (i%k) folds, not sklearn KFold contiguous blocks — diverges alpha_/coef_. kfold_indices in lasso_cv.rs must mirror sklearn check_cv(5)->KFold(5) non-shuffled contiguous splits (_coordinate_descent.py:1729). (#421)
- Divergence: ferrolearn_linear::FittedRidgeClassifier::predict binary boundary uses >=0 not >0 vs sklearn/linear_model/_base.py:384 (#405)
- Blocker for REQ-3 of ridge_cv: RidgeCV uses brute-force k-fold (default cv=5) over alphas grid; sklearn default cv=None uses _RidgeGCV efficient leave-one-out Generalized Cross-Validation (_ridge.py:2382-2412). Selected alpha_ diverges from sklearn default (#397)
- Translation unit: ferrolearn-linear/src/ridge_cv.rs (default LOO-GCV) (#403)
- Divergence: ferrolearn-linear Ridge::fit (alpha=0) errors on rank-deficient X where sklearn/_ridge.py:753 returns min-norm coef (#392)
- Divergence: ferrolearn-linear solve_lstsq diverges from sklearn/linear_model/_base.py:687 — rcond default eps vs max(m,n)*eps zeroes singular values scipy keeps (#381)
- Divergence: ferrolearn-linear solve_lstsq diverges from sklearn/linear_model/_base.py:687 — rcond default eps vs max(m,n)*eps zeroes singular values scipy keeps (#381)
- ferray-side (R-SUBSTRATE-5): ferray-linalg SVD precision on near-zero singular values diverges from LAPACK gelsd — s_min 5.0186e-15 vs 4.9735e-15 + different u_min, ~63% coef magnitude error on near-singular (cond~1e14) lstsq. Root: ferray-linalg/src/decomp/svd.rs. Blocks ferrolearn #381 (rcond fix Some(eps) is the ferrolearn-side half, lands with this). Fix in ferray's own vibe-fork harness. (#382)
- Divergence: ferrolearn-linear LinearRegression rejects valid underdetermined input (n_samples<n_features) with InsufficientSamples; sklearn succeeds (min-norm) (#377)
- Divergence: ferrolearn-linear LinearRegression rank-deficient X not minimum-norm (linalg::solve_lstsq QR vs sklearn gelsd SVD) (#376)
- datasets: add network fetch_* loaders + cache management (fetch_california_housing, get_data_home, clear_data_home, fetch_openml) (#321)
- numerical: scipy parity audit — special functions (gamma, beta, erf, etc.) + linalg (decompositions live in core::backend) (#322)
- model-sel: add make_pipeline, make_union helpers + threshold classifiers (FixedThresholdClassifier, TunedThresholdClassifierCV) (#316)
- model-sel: add inspection module (partial_dependence, permutation_importance) (#315)
- datasets: add file I/O loaders (load_svmlight_file, dump_svmlight_file, load_files) (#320)
- metrics: add scorer registry (get_scorer, get_scorer_names, check_scoring) + DistanceMetric trait (#308)
- model-sel: add ClassifierChain, RegressorChain, OutputCodeClassifier (#313)
- model-sel: add group-aware CV splitters (GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, LeavePGroupsOut, StratifiedGroupKFold) (#312)
- metrics: add ranking edge cases (coverage_error, label_ranking_average_precision_score, label_ranking_loss) (#307)
- metrics: add d2_* family (d2_absolute_error_score, d2_pinball_score, d2_tweedie_score, d2_brier_score, d2_log_loss_score) (#306)
- Add new crates for uncovered sklearn modules: covariance, neural_network (#252)
- covariance + neural: write api_proof.rs for both new crates (#328)
- neural: implement BernoulliRBM (Bernoulli-Bernoulli RBM with CD-1 training) (#327)
- neural: create ferrolearn-neural crate; move mlp.rs from linear; add to workspace + umbrella (#326)
- covariance: add GraphicalLasso + GraphicalLassoCV + function-style exports (empirical_covariance, ledoit_wolf, oas, shrunk_covariance, log_likelihood, fast_mcd) (#325)
- covariance: create ferrolearn-covariance crate; move covariance.rs from decomp; add to workspace + umbrella (#324)
- Audit utility crates (ferrolearn-core, ferrolearn-datasets, ferrolearn-sparse, ferrolearn-numerical, ferrolearn-io) vs sklearn equivalents (#251)
- utility crates: write tests/api_proof.rs for core, datasets, sparse, numerical, io (#323)
- sparse: add stack/eye/diags helpers (hstack, vstack, eye, diags, sparse_random) (#319)
- datasets: add 7 missing generators (make_friedman1/2/3, make_low_rank_matrix, make_spd_matrix, make_sparse_spd_matrix, make_gaussian_quantiles, make_hastie_10_2, make_multilabel_classification) (#318)
- Audit ferrolearn-model-sel vs sklearn (model_selection + pipeline + compose + multiclass + multioutput + dummy + frozen + inspection + calibration): close gaps, add API proof tests (#249)
- model-sel: write tests/api_proof.rs covering every public API (#317)
- model-sel: add dummy.rs (DummyClassifier, DummyRegressor) (#314)
- model-sel: add basic CV splitters (LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold, PredefinedSplit) (#311)
- model-sel: wire 4 orphaned modules (feature_union, multiclass, multioutput, transformed_target) in lib.rs (#310)
- Audit ferrolearn-metrics vs sklearn: close gaps, add API proof tests (#248)
- metrics: write tests/api_proof.rs covering every public API in ferrolearn-metrics (#309)
- metrics: add missing pairwise (pairwise_distances_argmin, argmin_min, pairwise_kernels) (#305)
- metrics: add missing clustering metrics (mutual_info_score, pair_confusion_matrix, homogeneity_completeness_v_measure, contingency_matrix) (#304)
- metrics: add 13 missing classification metrics (hamming, zero_one, balanced_accuracy, matthews_corrcoef, cohen_kappa, jaccard, fbeta, brier_score, hinge, multilabel_confusion_matrix, precision_recall_fscore_support, classification_report, det_curve) (#303)
- metrics: wire orphaned scorer module + 11 regression/clustering/pairwise re-exports in lib.rs (#302)
- Audit ferrolearn-preprocess vs sklearn (preprocessing + impute + feature_extraction + feature_selection): close gaps, add API proof tests (#247)
- Add proof-of-API integration test for ferrolearn-preprocess (#301)
- Wire orphaned preprocess estimators into lib.rs (LabelBinarizer, MultiLabelBinarizer, SelectFpr/Fdr/Fwe, SequentialFeatureSelector, feature scoring fns) (#299)
- Add GaussianRandomProjection / SparseRandomProjection / johnson_lindenstrauss_min_dim (#296)
- Audit ferrolearn-decomp vs sklearn (decomposition + cross_decomposition + manifold + random_projection): close gaps, add API proof tests (#246)
- Add proof-of-API integration test for ferrolearn-decomp (#298)
- Add inverse_transform to KernelPCA / IncrementalPCA / NMF / TruncatedSVD / FactorAnalysis (#295)
- Wire orphaned MiniBatchNMF and SparsePCA into ferrolearn-decomp lib.rs (#294)
- Audit ferrolearn-kernel vs sklearn (kernel_approximation + kernel_ridge): close gaps, add API proof tests (#250)
- Add proof-of-API integration test for ferrolearn-kernel (#292)
- Add sample_y() to GaussianProcessRegressor for posterior sampling (#291)
- Add predict_log_proba to GaussianProcessClassifier (#290)
- Add score() to KernelRidge / GaussianProcessRegressor / GaussianProcessClassifier (#289)
- Audit ferrolearn-linear vs sklearn (linear_model + svm + isotonic + discriminant_analysis): close gaps, add API proof tests (#245)
- Add proof-of-API integration test for ferrolearn-linear (#288)
- Add decision_function to LDA / QDA / RidgeClassifier / LogisticRegression / LogisticRegressionCV / LinearSVC / SGDClassifier (#287)
- Add predict_proba and predict_log_proba to classifiers missing them (LDA, QDA, RidgeClassifier, LogRegCV, SGDClassifier, LinearSVC) (#286)
- Add score() to every fitted linear / SVM / isotonic / discriminant_analysis estimator (#285)
- Wire 14 orphaned linear estimators into lib.rs (ARD, GLM family, Lars+LassoLars, LinearSVC/R, LogRegCV, OMP, QDA, QuantileRegressor, RidgeClassifier, MLP) (#284)
- Audit ferrolearn-cluster vs sklearn (cluster + mixture + semi_supervised): close gaps, add API proof tests (#244)
- Add proof-of-API integration test for ferrolearn-cluster (#282)
- Add predict_proba + score to LabelPropagation and LabelSpreading (#281)
- Add transform() to KMeans / MiniBatchKMeans / BisectingKMeans (#280)
- Fix GMM bic()/aic() signatures and add to BayesianGaussianMixture (#279)
- Add predict_proba, score, score_samples to GaussianMixture and BayesianGaussianMixture (#278)
- Add fit_predict and labels() accessor to all clustering estimators (#277)
- Audit ferrolearn-tree vs sklearn (tree + ensemble): close gaps, add API proof tests (#243)
- Add proof-of-API integration test for ferrolearn-tree (#275)
- Add decision_function to GradientBoosting / HistGradientBoosting / AdaBoost classifiers (#272)
- Add predict_log_proba to all classifiers (#270)
- Add predict_proba to remaining classifiers (RF, GB, HGB, AdaBoost, Bagging, Voting) (#269)
- Add feature_importances_ accessor to every tree-based estimator (#271)
- Add score() method to every fitted tree / ensemble estimator (#268)
- Wire orphaned modules into lib.rs: BaggingClassifier, BaggingRegressor, AdaBoostRegressor (#267)
- Audit ferrolearn-neighbors vs sklearn: close gaps, add API proof tests (#242)
- Add proof-of-API integration test for ferrolearn-neighbors (#266)
- Complete LocalOutlierFactor sklearn API: decision_function, fit_predict, score_samples, novelty mode (#265)
- Add kneighbors_graph and radius_neighbors_graph (free fns + methods) plus sort_graph_by_row_values (#264)
- Add kneighbors() and radius_neighbors() methods to supervised neighbors estimators (#263)
- Add score() method to all neighbors estimators (#262)
- Add predict_proba to KNeighborsClassifier and RadiusNeighborsClassifier (#261)
- Audit ferrolearn-bayes vs sklearn (naive_bayes + gaussian_process): close gaps, add API proof tests (#241)
- Add proof-of-API integration test exercising every public ferrolearn-bayes estimator end-to-end (#260)
- Add partial_fit method to CategoricalNB (#259)
- Add min_categories parameter to CategoricalNB (#258)
- Add norm parameter to ComplementNB (#257)
- Add force_alpha parameter to discrete Naive Bayes estimators (#256)
- Add fit_prior parameter to discrete Naive Bayes estimators (Multinomial, Bernoulli, Complement, Categorical) (#255)
- Add score() convenience method (mean accuracy) to all Naive Bayes fitted estimators (#254)
- Add predict_log_proba and predict_joint_log_proba methods to all Naive Bayes fitted estimators (#253)
- **ferrolearn-kernel**: GP-classifier prediction now uses Rasmussen & Williams Algorithm 3.2 — predictive variance via `K(x*, x*) − ‖L⁻¹√W K(x*, X)ᵀ‖²` and MacKay probit approximation `π̄* = σ(f̄*/√(1+πv*/8))` — replacing the prior shortcut that ignored predictive variance. Probability values are now better-calibrated for points far from training data (#237)
- **ferrolearn-numerical**: Replaced manual `(a + b) / 2.0` with `f64::midpoint(a, b)` in adaptive Simpson, Gauss-Kronrod, and cubic-spline routines for overflow-safe averaging (#239)

### Fixed
- translate(lda): REQ-7b priors validation — reject negative + renormalize sum!=1 (LDA differs from QDA) (#603)
- QuantileRegressor predictions 25x off from sklearn (IRLS vs HiGHS solver divergence) (#340)
- **ferrolearn-decomp**: `LLE::test_lle_different_n_neighbors` now asserts a real difference (`diff_sum > 1e-10`) instead of the no-op `diff_sum > 1e-10 || true` that always passed (#237)
- **ferrolearn-neighbors**: `test_all_algorithms_agree_kneighbors` now compares per-row sorted index sets across BruteForce/KdTree/BallTree, restoring an invariant that was previously dropped (the `reference_idxs` variable was assigned but never read) (#237)
- **ferrolearn-decomp** (`FittedPLSCanonical`, `FittedCCA`): removed stale `#[allow(dead_code)]` on `y_std_` field — it is in fact read by `transform_y` (#237)

### Maintenance
- Bumped 48 transitive dependency versions via `cargo update` (all patch-level, no breaking changes) (#237)
- Cleared 72 default-clippy warnings introduced by the rust 1.95 / clippy update (#238); remaining 67 auto-fixed via `cargo clippy --fix`
- Pedantic+nursery clippy: ~830 fixes across two passes — `redundant_closure`, `manual_let_else`, `single_match_else`, `uninlined_format_args`, `items_after_statements`, `explicit_iter_loop`, `cast_lossless`, `manual_midpoint`, `map_unwrap_or`, `option_if_let_else`, `semicolon_if_nothing_returned`, `ignored_unit_patterns`, `redundant_else`, `used_underscore_binding`, plus ~197 `or_fun_call` rewrites (`or_insert(F::zero())` → `or_insert_with(F::zero)`, `unwrap_or(F::epsilon())` → `unwrap_or_else(F::epsilon)`, etc.) (#239)
- 4 new GP classifier tests covering log-marginal-likelihood structural properties (finiteness, separability monotonicity, multiclass summation) and the new `classes()` accessor (#237)

### Added (post-0.1.0 features rolled into 0.2.2)
- Add RegressorChain for chained multi-target regression (#211)
- Add r_regression Pearson correlation for regression (#101)
- Add LassoLarsCV cross-validated LassoLars (#16)
- Add LeaveOneGroupOut and LeavePGroupsOut splitters (#159)
- Add AdditiveChi2Sampler for additive chi-squared kernel (#193)
- Add GraphicalLasso and GraphicalLassoCV sparse precision matrix (#202)
- Add StratifiedGroupKFold combined stratified+group split (#158)
- Add GroupShuffleSplit group-aware shuffle split (#157)
- Add PolynomialCountSketch for polynomial kernel (#195)
- Add LassoLarsIC Lasso with AIC/BIC selection (#17)
- Add PredefinedSplit for custom fold indices (#161)
- Add ClassifierChain for chained multi-label classification (#210)
- Add mutual_info_classif mutual information for classification (#99)
- Add OutputCodeClassifier error-correcting output codes (#206)
- Add mutual_info_regression mutual information for regression (#100)
- Add SkewedChi2Sampler for skewed chi-squared kernel (#194)
- Add LeavePOut exhaustive P-out cross-validation (#160)
- Expand oracle test coverage to 59 tests across 11 crates (28 new fixtures, 28 new tests)
- Add `brent_bounded` 1-D minimizer to ferrolearn-numerical (Brent's method with bounded interval)
- Add oracle tests for MultinomialNB, BernoulliNB, ComplementNB
- Add oracle tests for MiniBatchKMeans, MeanShift, GaussianMixture, OPTICS, Birch, SpectralClustering
- Add oracle tests for MaxAbsScaler, Normalizer, Binarizer, PolynomialFeatures, OneHotEncoder, LabelEncoder, QuantileTransformer, KBinsDiscretizer, SimpleImputer, PowerTransformer
- Add oracle tests for StratifiedKFold, TimeSeriesSplit
- Add oracle tests for ROC AUC, log loss, clustering metrics, extended regression metrics
- Add oracle tests for CubicSpline, statistical distributions, sparse eigendecomposition

### Fixed (post-0.1.0 fixes rolled into 0.2.2)
- Fix OPTICS Xi cluster extraction: rewrite to use steep-down areas with MIB tracking, region extension, and predecessor correction (matching sklearn's Figure 19 algorithm)
- Fix Birch final clustering: replace KMeans (naive init) with AgglomerativeClustering Ward linkage, eliminating initialization-dependent convergence failures
- Fix PowerTransformer lambda optimization: replace 201-point grid search (0.03 step) with Brent's method for continuous-precision optimization matching sklearn
- Fix StratifiedKFold remainder distribution: use round-robin fold offset across classes for balanced fold sizes (was front-loading extras to first folds)

## [0.1.0] - 2026-03-04

### Added
- Add missing scipy-equivalent numerical foundations (#19)
- Resolve open questions in kernel regression design document (#18)
- Add kernel regression design document for ferrolearn-kernel crate (#17)
- Add Pipeline support for f32 data (generic over float type) (#14)

Initial release with full scikit-learn-equivalent coverage across 14 crates.

### Phase 1: Foundation

- **ferrolearn-core**: `Fit`, `Predict`, `Transform`, `FitTransform` traits; `Dataset` type; `FerroError` error hierarchy; `Pipeline` with type-safe unfitted/fitted state; introspection traits (`HasCoefficients`, `HasFeatureImportances`, `HasClasses`)
- **ferrolearn-linear**: `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression` with L-BFGS optimizer
- **ferrolearn-preprocess**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`
- **ferrolearn-metrics**: Classification metrics (accuracy, precision, recall, F1, confusion matrix, ROC AUC, log loss); regression metrics (MAE, MSE, RMSE, R², MAPE)
- **ferrolearn-model-sel**: `KFold`, `StratifiedKFold`, `train_test_split`, `cross_val_score`
- **ferrolearn-sparse**: CSR, CSC, COO sparse matrix formats with conversions and arithmetic
- **ferrolearn-datasets**: `load_iris`, `load_diabetes`, `load_wine`; synthetic generators (`make_blobs`, `make_classification`, `make_regression`, `make_moons`, `make_circles`)
- Compile-fail tests ensuring unfitted models cannot call `predict()`
- Oracle test infrastructure with 10 sklearn fixture generators

### Phase 2: Classical ML

- **ferrolearn-tree**: `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor` with feature importances and configurable criteria
- **ferrolearn-neighbors**: `KNeighborsClassifier`, `KNeighborsRegressor` with KD-tree acceleration (auto-selected for dims <= 20) and distance weighting
- **ferrolearn-cluster**: `KMeans`, `DBSCAN`, `AgglomerativeClustering` (Ward/Complete/Average/Single linkage), `GaussianMixture`
- **ferrolearn-decomp**: `PCA`, `TruncatedSVD`, `NMF`, `KernelPCA` (RBF/polynomial/linear/sigmoid kernels)
- **ferrolearn-preprocess**: `SimpleImputer`, `VarianceThreshold`, `SelectKBest`, `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `PolynomialFeatures`
- **ferrolearn-io**: MessagePack and JSON model serialization with CRC32 integrity checks
- **ferrolearn-model-sel**: `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearch`, `param_grid!` macro, `TimeSeriesSplit`
- **ferrolearn-metrics**: Clustering metrics (`silhouette_score`, `adjusted_rand_index`, `normalized_mutual_info`, `calinski_harabasz`, `davies_bouldin`)

### Phase 3: Completeness

- **ferrolearn-tree**: `GradientBoostingClassifier`, `GradientBoostingRegressor` (least squares, LAD, Huber loss), `AdaBoostClassifier` (SAMME/SAMME.R)
- **ferrolearn-preprocess**: `MaxAbsScaler`, `Normalizer`, `Binarizer`, `PowerTransformer` (Yeo-Johnson/Box-Cox), `FunctionTransformer`
- **ferrolearn-core**: Compile-time type-safe `TypedPipeline`; pluggable `Backend` trait with `NdarrayFaerBackend` (gemm, svd, qr, cholesky, solve, eigh, det, inv)
- **ferrolearn-bayes**: `GaussianNB`, `MultinomialNB`, `BernoulliNB`, `ComplementNB`

### Phase 4: Beyond sklearn Baseline

- **ferrolearn-core**: `PartialFit` trait for online/incremental learning
- **ferrolearn-linear**: `ElasticNet`, `BayesianRidge`, `HuberRegressor`, `SGDClassifier`, `SGDRegressor`, `LDA` (Linear Discriminant Analysis)
- **ferrolearn-preprocess**: `ColumnTransformer`
- **ferrolearn-decomp**: `IncrementalPCA`, `FactorAnalysis`, `FastICA`, `Isomap`, `MDS`, `SpectralEmbedding`, `LLE`
- **ferrolearn-cluster**: `MiniBatchKMeans`, `MeanShift`, `SpectralClustering`, `OPTICS`
- **ferrolearn-model-sel**: `CalibratedClassifierCV`, `SelfTrainingClassifier`
- **ferrolearn-datasets**: `make_sparse_uncorrelated`

### Testing & Validation

- 1,468 tests across 14 crates, 0 failures
- 26 sklearn oracle tests comparing numerical output (predictions, coefficients, metrics) against scikit-learn 1.7.2 reference fixtures
- 7 end-to-end integration tests (classification pipeline, regression pipeline, clustering, cross-validation, serialization roundtrip, tree ensemble, preprocessing chain)
- Compile-fail tests for type-safety guarantees
- Fixture generation script (`scripts/generate_fixtures.py`) for reproducible sklearn baselines
