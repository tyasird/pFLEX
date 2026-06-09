# Archived legacy implementations retained for historical reference only.
# This module is intentionally not imported by pflex and is not public API.
# Active implementations live in analysis.py.

### OLD FUNCTIONS

# new but withoutparallel

# def pra_percomplex(dataset_name, matrix, is_corr=False):
#     log.started(f"*** Per-complex PRA started - {dataset_name} ***")
#     config = dload("config")
#     terms = dload("tmp", "terms")
#     genes_present = dload("tmp", "genes_present_in_terms")
#     sorting = dload("input", "sorting")
#     sort_order = sorting.get(dataset_name, "high")
#     if not is_corr:
#         matrix = perform_corr(matrix, config.get("corr_function"))
#     matrix = filter_matrix_by_genes(matrix, genes_present)
#     log.info(f"Matrix shape: {matrix.shape}")
#     df = binary(matrix)
#     log.info(f"Pair-wise shape: {df.shape}")
#     df = quick_sort(df, ascending=(sort_order == "low"))
#     pairwise_df = df.copy()
#     pairwise_df['gene1'] = pairwise_df['gene1'].astype("category")
#     pairwise_df['gene2'] = pairwise_df['gene2'].astype("category")
    
#     # Precompute a mapping from each gene to the row indices in the pairwise DataFrame where it appears.
#     gene_to_pair_indices = {}
#     for i, (gene_a, gene_b) in enumerate(zip(pairwise_df["gene1"], pairwise_df["gene2"])):
#         gene_to_pair_indices.setdefault(gene_a, []).append(i)
#         gene_to_pair_indices.setdefault(gene_b, []).append(i)
#     log.done
    
#     # Build gold_pair_to_complex using sets for efficiency
#     gold_pair_to_complex = defaultdict(set)
#     for idx, row in terms.iterrows():
#         genes = row.used_genes
#         if len(genes) < 2:
#             continue
#         for i, g1 in enumerate(genes):
#             for g2 in genes[i + 1:]:
#                 pair = tuple(sorted((g1, g2)))
#                 gold_pair_to_complex[pair].add(idx)
    
#     # Precompute complex_ids as semicolon-separated strings in pairwise_df
#     pairs = [tuple(sorted((g1, g2))) for g1, g2 in zip(pairwise_df["gene1"], pairwise_df["gene2"])]
#     pairwise_df['complex_ids'] = [';'.join(map(str, sorted(gold_pair_to_complex.get(pair, set())))) for pair in pairs]
    
#     # Initialize AUC scores
#     auc_scores = {}
#     # Loop over each gene complex
#     for idx, row in tqdm(terms.iterrows()):
#         gene_set = set(row.used_genes)
#         if config["min_genes_per_complex_analysis"] > len(gene_set):  
#             continue
#         # Collect all row indices in the pairwise data where either gene belongs to the complex.
#         candidate_indices = bitarray(len(pairwise_df))
#         for gene in gene_set:
#             if gene in gene_to_pair_indices:
#                 candidate_indices[gene_to_pair_indices[gene]] = True
        
#         if not candidate_indices.any():
#             continue
        
#         # Select only the relevant pairwise comparisons.
#         selected_rows = np.unpackbits(candidate_indices).view(bool)[:len(pairwise_df)]
#         sub_df = pairwise_df.iloc[selected_rows]
        
#         # Get current complex ID (assuming idx is the ID; adjust if row['ID'] is different)
#         complex_id = str(idx)  # Or str(row['ID']) if available
        
#         # Create true_label: 1 if complex_id in complex_ids (vectorized with str.contains)
#         #true_label = sub_df['complex_ids'].str.contains(complex_id, regex=False).astype(int)

#         # Inside the loop, for each complex:
#         # Inside the loop:
#         complex_id = str(idx)
#         # Use (?:^|;) and (?:;|$) to avoid capturing groups
#         pattern = r'(?:^|;)' + re.escape(complex_id) + r'(?:;|$)'
#         true_label = sub_df['complex_ids'].str.contains(pattern, regex=True).astype(int)
#         # Filter to keep verified negatives (complex_ids == "") or positives for this complex (true_label == 1)
#         complex_mask = (sub_df['complex_ids'] == "") | (true_label == 1)
        
#         # Use the masked true labels for AUPRC (avoids SettingWithCopyWarning)
#         predictions = true_label[complex_mask]
        
#         if predictions.sum() == 0:
#             continue
#         # Compute cumulative true positives and derive precision and recall.
#         true_positive_cumsum = predictions.cumsum()
#         precision = true_positive_cumsum / (np.arange(len(predictions)) + 1)
#         recall = true_positive_cumsum / true_positive_cumsum.iloc[-1]
        
#         if len(recall) < 2 or recall.iloc[-1] == 0:
#             continue
#         auc_scores[idx] = metrics.auc(recall, precision)
    
#     # Add the computed AUC scores to the terms DataFrame.
#     terms["auc_score"] = pd.Series(auc_scores)
#     terms.drop(columns=["hash"], inplace=True)
#     dsave(terms, "pra_percomplex", dataset_name)
#     log.done(f"Per-complex PRA completed.")
#     return terms

# it works quick but only maps 1 complex to each pair

# def pra_percomplex_old_type_filtering(dataset_name, matrix, is_corr=False):
#     log.started(f"*** Per-complex PRA started - {dataset_name} ***")
#     config = dload("config")
#     terms = dload("tmp", "terms")
#     genes_present = dload("tmp", "genes_present_in_terms")
#     sorting = dload("input", "sorting")
#     sort_order = sorting.get(dataset_name, "high")
#     if not is_corr:
#         matrix = perform_corr(matrix, config.get("corr_function"))
#     matrix = filter_matrix_by_genes(matrix, genes_present)
#     log.info(f"Matrix shape: {matrix.shape}")
#     df = binary(matrix)
#     log.info(f"Pair-wise shape: {df.shape}")
#     df = quick_sort(df, ascending=(sort_order == "low"))
#     pairwise_df = df.copy()
#     pairwise_df['gene1'] = pairwise_df['gene1'].astype("category")
#     pairwise_df['gene2'] = pairwise_df['gene2'].astype("category")  
#     # Precompute a mapping from each gene to the row indices in the pairwise DataFrame where it appears.
#     gene_to_pair_indices = {}
#     for i, (gene_a, gene_b) in enumerate(zip(pairwise_df["gene1"], pairwise_df["gene2"])):
#         gene_to_pair_indices.setdefault(gene_a, []).append(i)
#         gene_to_pair_indices.setdefault(gene_b, []).append(i)  
#     # Initialize AUC scores (one for each complex) with NaNs.
#     #auc_scores = np.full(len(terms), np.nan)
#     auc_scores = {}
#     # Loop over each gene complex
#     for idx, row in tqdm(terms.iterrows()):
#         gene_set = set(row.used_genes)

#         if config["min_genes_per_complex_analysis"] > len(gene_set):  
#             continue
#         # Collect all row indices in the pairwise data where either gene belongs to the complex.
#         candidate_indices = bitarray(len(pairwise_df))
#         for gene in gene_set:
#             if gene in gene_to_pair_indices:
#                 candidate_indices[gene_to_pair_indices[gene]] = True      
#         if not candidate_indices.any():
#             continue     
#         # Select only the relevant pairwise comparisons.
#         selected_rows = np.unpackbits(candidate_indices).view(bool)[:len(pairwise_df)]
#         sub_df = pairwise_df.iloc[selected_rows]
#         # A prediction is 1 if both genes in the pair are in the complex; otherwise 0.
#         predictions = (sub_df["gene1"].isin(gene_set) & sub_df["gene2"].isin(gene_set)).astype(int)
#         if predictions.sum() == 0:
#             continue
#         # Compute cumulative true positives and derive precision and recall.
#         true_positive_cumsum = predictions.cumsum()
#         precision = true_positive_cumsum / (np.arange(len(predictions)) + 1)
#         recall = true_positive_cumsum / true_positive_cumsum.iloc[-1]
        
#         if len(recall) < 2 or recall.iloc[-1] == 0:
#             continue
#         auc_scores[idx] = metrics.auc(recall, precision)   
#     # Add the computed AUC scores to the terms DataFrame.
#     terms["auc_score"] = pd.Series(auc_scores)
#     terms.drop(columns=["hash"], inplace=True)
#     dsave(terms, "pra_percomplex", dataset_name)
#     log.done(f"Per-complex PRA completed.")
#     return terms

# OLD
# def pra_percomplex(dataset_name, matrix, is_corr=False):
#     log.started(f"*** Per-complex PRA started for {dataset_name} ***")
#     config = dload("config")
#     terms = dload("tmp", "terms")
#     genes_present = dload("tmp", "genes_present_in_terms")
#     sorting = dload("input", "sorting")
#     sort_order = sorting.get(dataset_name, "high")

#     if not is_corr:
#         matrix = perform_corr(matrix, "numpy")
#     matrix = filter_matrix_by_genes(matrix, genes_present)
#     log.info(f"Matrix shape: {matrix.shape}")
#     df = binary(matrix)
#     log.info(f"Pair-wise shape: {df.shape}")
#     df = quick_sort(df, ascending=(sort_order == "low"))
#     # Precompute gene → row indices
#     gene_to_rows = {}
#     for i, (g1, g2) in enumerate(zip(df["gene1"], df["gene2"])):
#         gene_to_rows.setdefault(g1, []).append(i)
#         gene_to_rows.setdefault(g2, []).append(i)
#     aucs = np.full(len(terms), np.nan)
#     N = len(df)
#     for idx, row in tqdm(terms.iterrows()):
#         genes = set(row.used_genes)
#         if len(genes) < config["min_complex_size_for_percomplex"]:  # Skip small complexes
#             continue
#         # Get all row indices where either gene is in the complex
#         candidate_idxs = set()
#         for g in genes:
#             candidate_idxs.update(gene_to_rows.get(g, []))
#         candidate_idxs = sorted(candidate_idxs)
#         if not candidate_idxs:
#             continue
#         # Use only relevant rows for prediction
#         sub = df.loc[candidate_idxs]
#         preds = (sub["gene1"].isin(genes) & sub["gene2"].isin(genes)).astype(int)
#         if preds.sum() == 0:
#             continue
#         tp = preds.cumsum()
#         prec = tp / (np.arange(len(preds)) + 1)
#         recall = tp / tp.iloc[-1]
#         if len(recall) < 2 or recall.iloc[-1] == 0:
#             continue
#         aucs[idx] = metrics.auc(recall, prec)
#     terms["auc_score"] = aucs
#     terms.drop(columns=["list", "set", "hash"], inplace=True)
#     dsave(terms, "pra_percomplex", dataset_name)
#     log.done(f"Per-complex PRA completed.")
#     return terms

# without greedy
# def complex_contributions(name):
#     log.info(f"Computing complex contributions for dataset: {name}")

#     pra = dload("pra", name)
#     terms = dload("tmp", "terms")
#     d = pra.query('prediction == 1').drop(columns=['gene1', 'gene2'])
#     results = {}
#     thresholds = [round(i, 2) for i in np.arange(1, 0.0001, -0.025)]
#     for cid in terms.ID.to_list():
#         arr = []
#         for threshold in thresholds:
#             r = d[d.complex_id == cid].query('precision >= @threshold')
#             arr.append(r.shape[0])
#         results[cid] = arr

#     r = pd.DataFrame(results, index=thresholds).T
#     t = terms[['ID', 'Name']].set_index('ID')
#     r['Name'] = r.index.map(t.Name)
#     r = r[list(reversed(list(r.columns)))]
#     r = r.reset_index(drop=True)
#     dsave(r, "complex_contributions", name)
#     log.info(f"Complex contributions computation completed for dataset: {name}")
#     return r

# # new
# def complex_contributions(name):
#     log.info(f"Computing complex contributions using R-style greedy logic for dataset: {name}")
#     pra = dload("pra", name)
#     terms = dload("common", "terms")
    
#     # Ensure pra is sorted by score descending
#     pra = pra.sort_values(by='score', ascending=False).reset_index(drop=True)
    
#     # Compute cumulative TP and precision if not present
#     pra['cumTP'] = pra['prediction'].cumsum()
#     pra['rank'] = pra.index + 1
#     pra['precision'] = pra['cumTP'] / pra['rank']
    
#     # R-style precision thresholds
#     prec_min = pra['precision'].min()
#     prec_max = pra['precision'].max()
#     precision_cutoffs = [round(prec_min, 3)]
#     cutoffs_range = np.arange(0.1, prec_max + 0.001, 0.025)
#     precision_cutoffs += [round(t, 3) for t in cutoffs_range if t > prec_min]
#     thresholds = sorted(set(precision_cutoffs))  # Ensure unique and sorted
    
#     results = {}
#     for t in thresholds:
#         if pra['precision'].max() < t:
#             continue
#         cand = pra[pra['precision'] >= t]
#         if cand.empty:
#             continue
#         k = cand.index.max()  # rightmost index where precision >= t
#         tp_target = pra.loc[k, 'cumTP']
#         # Find the smallest m where cumTP[m] >= tp_target
#         ind = pra[pra['cumTP'] >= tp_target].index.min()
#         if pd.isna(ind):
#             continue
#         # Select top (ind+1) rows
#         tmp = pra.iloc[0:ind + 1].copy()
#         # Filter for predicted positives (true == 1)
#         tmp = tmp[tmp['prediction'] == 1]
#         tmp = tmp[tmp["complex_id"].notnull()]
#         tmp["ID"] = tmp["complex_id"].apply(lambda ids: ";".join(str(int(i)) for i in ids if pd.notnull(i)))
#         # Now greedy logic
#         final_contrib = {}
#         while not tmp.empty:
#             all_ids = tmp["ID"].str.split(";").explode()
#             contrib = all_ids.value_counts()
#             if contrib.empty:
#                 break
#             top_id = contrib.idxmax()
#             final_contrib[top_id] = contrib[top_id]
#             tmp = tmp[~tmp["ID"].str.contains(rf"\b{top_id}\b", regex=True)]
#         for cid, count in final_contrib.items():
#             if cid not in results:
#                 results[cid] = [0] * len(thresholds)
#             results[cid][thresholds.index(t)] = count
    
#     # Add back gold standard complexes with 0 contribution
#     gold_ids = set(terms.index.astype(str))
#     all_ids = set(results.keys())
#     missing_ids = gold_ids - all_ids
#     for cid in missing_ids:
#         results[cid] = [0] * len(thresholds)
    
#     # Build result DataFrame
#     r = pd.DataFrame(results, index=thresholds).T
#     r['Name'] = r.index.astype(int).map(terms['Name'])
#     r = r[['Name'] + [c for c in r.columns if c != 'Name']]  # Name as first col
#     r = r[(r.drop(columns="Name").sum(axis=1) > 0)]
#     # Move ID to first column, keep Name second, then precision columns in order
#     dsave(r, "complex_contributions", name)
#     log.info(f"Greedy R-style complex contribution completed for dataset: {name}")
#     return r

# def pra(dataset_name, matrix, is_corr=False):
#     log.info(f"******************** {dataset_name} ********************")
#     log.started(f"** Global Precision-Recall Analysis - {dataset_name} **")
#     config = dload("config")

#     terms_data = dload("tmp", "terms")
#     if terms_data is None or not isinstance(terms_data, pd.DataFrame):
#         raise ValueError("Expected 'terms' to be a DataFrame, but got None or invalid type.")
#     terms = terms_data
#     genes_present = dload("tmp", "genes_present_in_terms")
#     sorting = dload("input", "sorting")
#     sort_order = sorting.get(dataset_name, "high")

#     if not is_corr:
#         matrix = perform_corr(matrix, config.get("corr_function"))
        
#     matrix = filter_matrix_by_genes(matrix, genes_present)

#     log.info(f"Matrix shape: {matrix.shape}")
#     df = binary(matrix)
#     log.info(f"Pair-wise shape: {df.shape}")
#     df = quick_sort(df, ascending=(sort_order == "low"))

#     gold_pair_to_complex = defaultdict(list)
#     for idx, row in terms.iterrows():
#         genes = row.used_genes
#         if len(genes) < 2:
#             continue
#         for i, g1 in enumerate(genes):
#             for g2 in genes[i + 1:]:
#                 pair = tuple(sorted((g1, g2)))
#                 gold_pair_to_complex[pair].append(idx)

#     # Label predictions and complex IDs
#     complex_ids = []
#     predictions = []
#     for g1, g2 in zip(df["gene1"], df["gene2"]):
#         pair = tuple(sorted((g1, g2)))
#         ids = gold_pair_to_complex.get(pair, [])
#         if ids:
#             predictions.append(1)
#             complex_ids.append(ids)
#         else:
#             predictions.append(0)
#             complex_ids.append([])

#     df["prediction"] = predictions
#     df["complex_id"] = complex_ids

#     if df["prediction"].sum() == 0:
#         log.info("No true positives found in dataset.")
#         pr_auc = np.nan
#     else:
#         tp = df["prediction"].cumsum()
#         df["tp"] = tp
#         precision = tp / (np.arange(len(df)) + 1)
#         recall = tp / tp.iloc[-1]
#         pr_auc = metrics.auc(recall, precision)
#         df["precision"] = precision
#         df["recall"] = recall

#     log.info(f"PR-AUC: {pr_auc:.4f}, Number of true positives: {df['prediction'].sum()}")
#     dsave(df, "pra", dataset_name)
#     dsave(pr_auc, "pr_auc", dataset_name)
#     log.done(f"Global PRA completed for {dataset_name}")
#     return df, pr_auc

# def compute_pra(df):
#     log.info("Calculating precision-recall and AUC score.")
#     if df.empty:
#         log.warning("Empty DataFrame encountered in compute_pra. Returning empty DataFrame.")
#         return df  
#     df["tp"] = df["prediction"].cumsum()
#     df.reset_index(drop=True, inplace=True)
#     df["precision"] = df["tp"] / (df.index + 1)
#     df["recall"] = df["tp"] / df["tp"].iloc[-1]
#     log.info("DONE: Calculating precision-recall AUC score.")
#     return df

# def pra(dataset_name, matrix, is_corr=False):
#     log.info(f"PRA computation started for {dataset_name}.")
#     genes_present_in_terms = dload("tmp", "genes_present_in_terms")
#     #terms_hash_table = dload("tmp", "terms_hash_table")
#     sorting_prefs = dload("input", "sorting")
#     sort_order = sorting_prefs.get(dataset_name, "high") 
#     if not is_corr: matrix = perform_corr(matrix, "numpy")
#     matrix = filter_matrix_by_genes(matrix, genes_present_in_terms)
#     stack = binary(matrix)

#     log.info("Checking gene pairs against the gold standard.")
#     gene_pairs = list(zip(stack["gene1"], stack["gene2"]))
#     hashed_pairs = [hash(pair) for pair in gene_pairs]
#     stack["complex_id"] = [terms_hash_table.get(h, 0) for h in hashed_pairs]
#     stack["prediction"] = [1 if h in terms_hash_table else 0 for h in hashed_pairs]

#     annotated = stack.copy()
#     if sort_order == "low":
#         ann_sorted = quick_sort(annotated, ascending=True) 
#     else:
#         ann_sorted = quick_sort(annotated) 

#     pra = compute_pra(ann_sorted)
#     pr_auc = metrics.auc(pra.recall, pra.precision)
#     dsave(pra, "pra", dataset_name)
#     dsave(pr_auc, "pr_auc", dataset_name)
#     log.info(f"PRA computation completed for {dataset_name} (Sorting: {sort_order}).")
#     return pra, pr_auc
