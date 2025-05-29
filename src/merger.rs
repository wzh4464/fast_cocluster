use std::collections::{BinaryHeap, HashSet, HashMap};
use std::cmp::Ordering;
use ndarray::Array2;

// Placeholder for SubmatrixWithScore, adjust as needed
#[derive(Debug, Clone)]
pub struct SubmatrixWithScore {
    pub id: usize, // Unique identifier for the submatrix
    pub row_indices: HashSet<usize>,
    pub col_indices: HashSet<usize>,
    pub score: f64,
    pub coherence: f64,
    pub density: f64,
    pub size: f64,
    // Add other relevant fields, e.g., the actual data or reference to it
}

// Implement Eq, PartialEq, Ord, PartialOrd for BinaryHeap
impl PartialEq for SubmatrixWithScore {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for SubmatrixWithScore {}

impl PartialOrd for SubmatrixWithScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Order by score descending for Max-Heap behavior in BinaryHeap
        // If scores are equal, you might want a tie-breaking rule, e.g., by id
        other.score.partial_cmp(&self.score) 
    }
}

impl Ord for SubmatrixWithScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone)]
pub struct HierarchicalMerger {
    pub overlap_threshold: f64,
    pub score_weights: (f64, f64, f64), // (coherence, density, size)
}

impl HierarchicalMerger {
    pub fn new(overlap_threshold: f64, coherence_weight: f64, 
               density_weight: f64, size_weight: f64) -> Self {
        Self {
            overlap_threshold,
            score_weights: (coherence_weight, density_weight, size_weight),
        }
    }

    pub fn merge(&self, local_results: Vec<Vec<SubmatrixWithScore>>, 
                 original_matrix: &Array2<f64>) -> Vec<SubmatrixWithScore> {
        let mut all_submatrices = Vec::new();
        let mut id_counter = 0;
        
        for result_group in local_results {
            for mut submatrix in result_group {
                // Assign a unique ID here if not already consistently assigned
                // The provided code for DiMergeCo.local_cocluster already assigns an ID.
                // If IDs from local_cocluster are not globally unique, re-assign here.
                // For now, assuming IDs from local_cocluster might overlap between blocks, so re-assigning.
                submatrix.id = id_counter;
                id_counter += 1;
                all_submatrices.push(submatrix);
            }
        }
        
        if all_submatrices.is_empty() {
            return Vec::new();
        }

        // Binary tree-based hierarchical merging
        let mut current_level: Vec<SubmatrixWithScore> = all_submatrices;
        
        loop { // Loop until no more merges are made in a pass
            let mut next_level = Vec::new();
            let mut merged_in_pass = false;
            let mut processed_indices_in_current_level = HashSet::new(); // Tracks indices in current_level
            
            // Sort by score in descending order to prioritize better submatrices
            current_level.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            
            for i in 0..current_level.len() {
                if processed_indices_in_current_level.contains(&i) {
                    continue;
                }
                
                let current_submatrix = &current_level[i];
                let mut best_merge_candidate_info: Option<(usize, SubmatrixWithScore)> = None;
                // let mut best_score_after_merge = current_submatrix.score; // Current submatrix's score is baseline

                for j in (i + 1)..current_level.len() {
                    if processed_indices_in_current_level.contains(&j) {
                        continue;
                    }
                    
                    let candidate_submatrix = &current_level[j];
                    let overlap = self.compute_overlap_ratio(current_submatrix, candidate_submatrix);
                    
                    if overlap >= self.overlap_threshold {
                        // Try merging and see if the new score is better
                        let temp_merged = self.merge_submatrices(current_submatrix, candidate_submatrix, 
                                                               id_counter, original_matrix); // Tentative ID
                        
                        // Only consider merging if the merged score is better than both individual scores (or configurable policy)
                        // Or if it's better than the current best_merge_candidate_info's score
                        let current_best_score_for_merge = match &best_merge_candidate_info {
                            Some((_, sm_cand)) => sm_cand.score,
                            None => current_submatrix.score.max(candidate_submatrix.score), // Compare against combining current with another
                        };

                        if temp_merged.score > current_best_score_for_merge && temp_merged.score > current_submatrix.score && temp_merged.score > candidate_submatrix.score {
                             best_merge_candidate_info = Some((j, temp_merged));
                            // best_score_after_merge = temp_merged.score;
                        }
                    }
                }
                
                if let Some((merged_with_idx, merged_submatrix)) = best_merge_candidate_info {
                    processed_indices_in_current_level.insert(i); // Mark current as processed for this pass
                    processed_indices_in_current_level.insert(merged_with_idx); // Mark the one it merged with
                    next_level.push(merged_submatrix); // This already has a new id from merge_submatrices
                    id_counter +=1; // Increment for the next unique ID
                    merged_in_pass = true;
                } else {
                    // If current_submatrix was not merged with anything, it moves to the next level as is.
                    // Only add if not already processed (e.g. as part of a previous merge in this pass)
                     if !processed_indices_in_current_level.contains(&i) {
                        next_level.push(current_submatrix.clone());
                        processed_indices_in_current_level.insert(i); // Mark as processed for this pass
                    }
                }
            }
             // Add any remaining unprocessed items from current_level to next_level
            // This handles cases where items at the end of the sorted list weren't considered as `current_submatrix`
            // or were not merged.
            for i in 0..current_level.len() {
                if !processed_indices_in_current_level.contains(&i) {
                    next_level.push(current_level[i].clone());
                }
            }
            
            if !merged_in_pass || next_level.len() <= 1 { // Convergence or only one item left
                current_level = next_level; // Final set of merged submatrices
                break;
            }
            
            current_level = next_level;
        }
        
        current_level
    }
    
    fn compute_overlap_ratio(&self, a: &SubmatrixWithScore, b: &SubmatrixWithScore) -> f64 {
        let row_intersection = a.row_indices.intersection(&b.row_indices).count();
        let col_intersection = a.col_indices.intersection(&b.col_indices).count();
        
        let row_union = a.row_indices.union(&b.row_indices).count();
        let col_union = a.col_indices.union(&b.col_indices).count();
        
        if row_union == 0 || col_union == 0 { // Avoid division by zero
            return 0.0;
        }
        
        let intersection_size = row_intersection * col_intersection;
        let union_size = row_union * col_union; // Total cells in the union of bounding boxes
        
        if union_size == 0 { return 0.0; }
        intersection_size as f64 / union_size as f64
    }
    
    pub fn merge_submatrices(&self, a: &SubmatrixWithScore, b: &SubmatrixWithScore, 
                        new_id: usize, matrix: &Array2<f64>) -> SubmatrixWithScore {
        // Merge row and column indices
        let mut row_indices = a.row_indices.clone();
        row_indices.extend(b.row_indices.iter().cloned()); // Ensure correct extension
        
        let mut col_indices = a.col_indices.clone();
        col_indices.extend(b.col_indices.iter().cloned()); // Ensure correct extension
        
        // Compute new quality metrics
        let coherence = self.compute_coherence(&row_indices, &col_indices, matrix);
        let density = self.compute_density(&row_indices, &col_indices, matrix);
        let size_metric = self.compute_size_metric(&row_indices, &col_indices); // Changed from compute_size
        
        // Compute weighted score
        let score = self.score_weights.0 * coherence + 
                   self.score_weights.1 * density + 
                   self.score_weights.2 * size_metric;
        
        SubmatrixWithScore {
            id: new_id,
            row_indices,
            col_indices,
            score,
            coherence,
            density,
            size: size_metric, // Store the computed size metric here
        }
    }
    
    pub fn compute_coherence(&self, rows: &HashSet<usize>, cols: &HashSet<usize>, 
                        matrix: &Array2<f64>) -> f64 {
        if rows.is_empty() || cols.is_empty() {
            return 0.0;
        }
        let mut sum_sq = 0.0;
        let mut count = 0;
        
        for &row in rows {
            for &col in cols {
                // Check bounds, though HashSets should contain valid indices from original matrix context
                if row < matrix.nrows() && col < matrix.ncols() {
                    sum_sq += matrix[(row, col)].powi(2);
                    count += 1;
                } else {
                    // Handle out-of-bounds, though ideally indices are always valid
                }
            }
        }
        
        if count == 0 {
            return 0.0;
        }
        
        (sum_sq / count as f64).sqrt() // RMS value as coherence
    }
    
    pub fn compute_density(&self, rows: &HashSet<usize>, cols: &HashSet<usize>, 
                      matrix: &Array2<f64>) -> f64 {
        if rows.is_empty() || cols.is_empty() {
            return 0.0;
        }
        let mut sum_abs = 0.0; // Changed from sum to sum_abs based on common density definitions
        let mut count = 0;
        
        for &row in rows {
            for &col in cols {
                if row < matrix.nrows() && col < matrix.ncols() {
                    sum_abs += matrix[(row, col)].abs();
                    count += 1;
                }
            }
        }
        
        if count == 0 {
            return 0.0;
        }
        
        sum_abs / count as f64 // Average absolute value as density
    }
    
    // Renamed from compute_size to compute_size_metric to avoid confusion with field name
    pub fn compute_size_metric(&self, rows: &HashSet<usize>, cols: &HashSet<usize>) -> f64 {
        // A common size metric is harmonic mean or min dimension, or geometric mean
        // Using min dimension as per the user's previous code for `size` field
        if rows.is_empty() || cols.is_empty() {
            0.0
        } else {
            (rows.len().min(cols.len())) as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Imports HierarchicalMerger, SubmatrixWithScore, etc.
    use ndarray::array; 
    use std::collections::HashSet;

    // Helper to create a SubmatrixWithScore for testing
    fn create_test_submatrix(id: usize, rows: &[usize], cols: &[usize], score: f64, 
                             coherence: f64, density: f64, size: f64) -> SubmatrixWithScore {
        SubmatrixWithScore {
            id,
            row_indices: rows.iter().cloned().collect(),
            col_indices: cols.iter().cloned().collect(),
            score,
            coherence,
            density,
            size,
        }
    }

    #[test]
    fn test_submatrix_ordering() {
        let sm1 = create_test_submatrix(1, &[0], &[0], 0.8, 0.0, 0.0, 0.0);
        let sm2 = create_test_submatrix(2, &[1], &[1], 0.9, 0.0, 0.0, 0.0); // Higher score
        let sm3 = create_test_submatrix(3, &[2], &[2], 0.8, 0.0, 0.0, 0.0); // Same score as sm1

        let mut heap = BinaryHeap::new();
        heap.push(sm1.clone());
        heap.push(sm2.clone());
        heap.push(sm3.clone());

        assert_eq!(heap.pop().unwrap().id, 2); // sm2 should be popped first (highest score)
        // The order of sm1 and sm3 might vary as their scores are equal, depends on internal tie-breaking
        // or insertion order if tie-breaking is not strict via id. BinaryHeap is not guaranteed stable.
        let next_pop_id = heap.pop().unwrap().id;
        assert!(next_pop_id == 1 || next_pop_id == 3);
        let last_pop_id = heap.pop().unwrap().id;
        assert!(last_pop_id == 1 || last_pop_id == 3);
        assert_ne!(next_pop_id, last_pop_id);
    }

    #[test]
    fn test_compute_overlap_ratio() {
        let merger = HierarchicalMerger::new(0.5, 1.0, 1.0, 1.0);
        let sm1 = create_test_submatrix(1, &[0, 1], &[0, 1], 0.0,0.0,0.0,0.0); // 2x2
        let sm2 = create_test_submatrix(2, &[1, 2], &[1, 2], 0.0,0.0,0.0,0.0); // 2x2, overlaps 1x1
        // Intersection: rows {1}, cols {1} -> 1x1 = 1 cell
        // Union: rows {0,1,2}, cols {0,1,2} -> 3x3 = 9 cells
        let overlap12 = merger.compute_overlap_ratio(&sm1, &sm2);
        assert!((overlap12 - (1.0 * 1.0) / (3.0 * 3.0)).abs() < 1e-9, "Overlap 1-2 error, got: {}", overlap12);

        let sm3 = create_test_submatrix(3, &[0], &[0], 0.0,0.0,0.0,0.0); // 1x1
        let sm4 = create_test_submatrix(4, &[10], &[10], 0.0,0.0,0.0,0.0); // 1x1, no overlap
        let overlap34 = merger.compute_overlap_ratio(&sm3, &sm4);
        assert!((overlap34 - 0.0).abs() < 1e-9, "Overlap 3-4 error, got: {}", overlap34);

        let sm5 = create_test_submatrix(5, &[0, 1], &[0, 1], 0.0,0.0,0.0,0.0); // 2x2
        let sm6 = create_test_submatrix(6, &[0], &[0], 0.0,0.0,0.0,0.0); // 1x1, fully contained
        // Intersection: rows {0}, cols {0} -> 1x1 = 1 cell
        // Union: rows {0,1}, cols {0,1} -> 2x2 = 4 cells
        let overlap56 = merger.compute_overlap_ratio(&sm5, &sm6);
        assert!((overlap56 - (1.0*1.0) / (2.0*2.0)).abs() < 1e-9, "Overlap 5-6 error, got: {}", overlap56);
        
        let sm_empty_rows = create_test_submatrix(7, &[], &[0,1], 0.0,0.0,0.0,0.0);
        let sm_empty_cols = create_test_submatrix(8, &[0,1], &[], 0.0,0.0,0.0,0.0);
        let overlap_empty1 = merger.compute_overlap_ratio(&sm1, &sm_empty_rows);
        assert_eq!(overlap_empty1, 0.0);
        let overlap_empty2 = merger.compute_overlap_ratio(&sm1, &sm_empty_cols);
        assert_eq!(overlap_empty2, 0.0);
        let overlap_empty_both = merger.compute_overlap_ratio(&sm_empty_rows, &sm_empty_cols);
        assert_eq!(overlap_empty_both, 0.0);
    }

    #[test]
    fn test_compute_quality_metrics() {
        let merger = HierarchicalMerger::new(0.5, 1.0, 1.0, 1.0);
        let matrix = array![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
        
        let rows1: HashSet<usize> = [0, 1].iter().cloned().collect();
        let cols1: HashSet<usize> = [0, 1].iter().cloned().collect();
        // Submatrix: [[1,2],[3,4]]
        // Coherence: sqrt((1^2+2^2+3^2+4^2)/4) = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) approx 2.7386
        let coherence1 = merger.compute_coherence(&rows1, &cols1, &matrix);
        assert!((coherence1 - (30.0f64 / 4.0).sqrt()).abs() < 1e-9, "Coherence 1 error, got: {}", coherence1);
        // Density: (1+2+3+4)/4 = 10/4 = 2.5
        let density1 = merger.compute_density(&rows1, &cols1, &matrix);
        assert!((density1 - 2.5).abs() < 1e-9, "Density 1 error, got: {}", density1);
        // Size: Use the current definition, e.g. min(rows.len(), cols.len())
        // The previous test had log2(rows.len()*cols.len()), but compute_size_metric is min dimension
        let size1 = merger.compute_size_metric(&rows1, &cols1); // Removed &matrix
        assert!((size1 - (rows1.len().min(cols1.len())) as f64).abs() < 1e-9, "Size 1 error. Expected {}, got {}", (rows1.len().min(cols1.len())) as f64, size1);

        let rows_empty: HashSet<usize> = HashSet::new();
        let cols_empty: HashSet<usize> = HashSet::new();
        assert_eq!(merger.compute_coherence(&rows_empty, &cols1, &matrix), 0.0);
        assert_eq!(merger.compute_density(&rows1, &cols_empty, &matrix), 0.0);
        assert_eq!(merger.compute_size_metric(&rows_empty, &cols_empty), 0.0);
    }

    #[test]
    fn test_merge_submatrices_basic() {
        let merger = HierarchicalMerger::new(0.1, 0.3, 0.4, 0.3); // weights (coh, den, siz)
        let matrix = array![[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let sm1 = create_test_submatrix(1, &[0], &[0], 0.5, 1.0, 1.0, 0.0); // 1x1 at (0,0), val 1.0. size log2(1)=0
        let sm2 = create_test_submatrix(2, &[1], &[1], 0.5, 1.0, 1.0, 0.0); // 1x1 at (1,1), val 1.0. size log2(1)=0

        let merged = merger.merge_submatrices(&sm1, &sm2, 3, &matrix);
        // Merged indices: rows {0,1}, cols {0,1}. Submatrix: [[1,1],[1,1]]
        // Coherence: sqrt((1*1+1*1+1*1+1*1)/4) = sqrt(4/4) = 1.0
        // Density: (1+1+1+1)/4 = 1.0
        // Size: log2(2*2) = 2.0
        // Score: 0.3*1.0 + 0.4*1.0 + 0.3*2.0 = 0.3 + 0.4 + 0.6 = 1.3
        assert_eq!(merged.id, 3);
        assert_eq!(merged.row_indices, [0, 1].iter().cloned().collect());
        assert_eq!(merged.col_indices, [0, 1].iter().cloned().collect());
        assert!((merged.coherence - 1.0).abs() < 1e-9);
        assert!((merged.density - 1.0).abs() < 1e-9);
        assert!((merged.size - 2.0).abs() < 1e-9);
        assert!((merged.score - 1.3).abs() < 1e-9, "Merged score error, got: {}", merged.score);
    }

    #[test]
    fn test_hierarchical_merger_no_merge() {
        let merger = HierarchicalMerger::new(0.8, 1.0, 1.0, 1.0); // High overlap threshold
        let matrix = Array2::<f64>::zeros((2,2));
        let sm1 = create_test_submatrix(1, &[0], &[0], 0.9, 1.0,1.0,1.0);
        let sm2 = create_test_submatrix(2, &[1], &[1], 0.8, 1.0,1.0,1.0);
        let local_results = vec![vec![sm1.clone(), sm2.clone()]];

        let result = merger.merge(local_results, &matrix);
        assert_eq!(result.len(), 2, "Expected 2 submatrices, no merge");
        // Order might change due to sorting inside merge, check ids
        assert!(result.iter().any(|sm| sm.id == 1));
        assert!(result.iter().any(|sm| sm.id == 2));
    }

    #[test]
    fn test_hierarchical_merger_simple_merge() {
        // Overlap is high, scores are designed so merged is better
        let merger = HierarchicalMerger::new(0.1, 0.3, 0.4, 0.3); // low threshold
        let matrix = array![[10.0, 10.0, 0.0], [10.0, 10.0, 0.0], [0.0, 0.0, 1.0]];

        // sm1: [[10]] at (0,0). Coherence=10, Density=10, Size=0. Score: 0.3*10+0.4*10+0.3*0 = 3+4=7
        let sm1_rows: HashSet<usize> = [0].iter().cloned().collect();
        let sm1_cols: HashSet<usize> = [0].iter().cloned().collect();
        let sm1_coh = merger.compute_coherence(&sm1_rows, &sm1_cols, &matrix);
        let sm1_den = merger.compute_density(&sm1_rows, &sm1_cols, &matrix);
        let sm1_siz = merger.compute_size_metric(&sm1_rows, &sm1_cols);
        let sm1_sco = merger.score_weights.0 * sm1_coh + merger.score_weights.1 * sm1_den + merger.score_weights.2 * sm1_siz;
        let sm1 = create_test_submatrix(1, &[0], &[0], sm1_sco, sm1_coh, sm1_den, sm1_siz);
        assert!((sm1_sco - 7.0).abs() < 1e-9, "sm1 score mismatch {}", sm1_sco);

        // sm2: [[10]] at (1,1). Coherence=10, Density=10, Size=0. Score = 7
        let sm2_rows: HashSet<usize> = [1].iter().cloned().collect();
        let sm2_cols: HashSet<usize> = [1].iter().cloned().collect();
        let sm2_coh = merger.compute_coherence(&sm2_rows, &sm2_cols, &matrix);
        let sm2_den = merger.compute_density(&sm2_rows, &sm2_cols, &matrix);
        let sm2_siz = merger.compute_size_metric(&sm2_rows, &sm2_cols);
        let sm2_sco = merger.score_weights.0 * sm2_coh + merger.score_weights.1 * sm2_den + merger.score_weights.2 * sm2_siz;
        let sm2 = create_test_submatrix(2, &[1], &[1], sm2_sco, sm2_coh, sm2_den, sm2_siz);
        assert!((sm2_sco - 7.0).abs() < 1e-9, "sm2 score mismatch {}", sm2_sco);
        
        // Merged: rows {0,1}, cols {0,1}. Matrix [[10,10],[10,10]]
        // Coherence = 10, Density = 10, Size = log2(4) = 2
        // Score = 0.3*10 + 0.4*10 + 0.3*2 = 3+4+0.6 = 7.6. This is > 7.
        let local_results = vec![vec![sm1, sm2]];
        let result = merger.merge(local_results, &matrix);
        
        assert_eq!(result.len(), 1, "Expected 1 merged submatrix");
        let merged_sm = &result[0];
        assert!((merged_sm.score - 7.6).abs() < 1e-9, "Final merged score error, got: {}", merged_sm.score);
        assert_eq!(merged_sm.row_indices.len(), 2);
        assert_eq!(merged_sm.col_indices.len(), 2);
    }

    #[test]
    fn test_hierarchical_merger_multiple_levels() {
        let merger = HierarchicalMerger::new(0.05, 0.3, 0.4, 0.3); // Very low threshold
        let matrix = array![
            [10.0, 10.0,  0.0,  0.0],
            [10.0, 10.0,  0.0,  0.0],
            [ 0.0,  0.0, 10.0, 10.0],
            [ 0.0,  0.0, 10.0, 10.0]
        ];
        // Create 4 small submatrices that should merge into one 2x2, then those two 2x2s merge into one 4x4 (conceptually)
        // For simplicity, we make them such that they merge pairwise first.

        let sm_a_r: HashSet<usize> = [0].iter().cloned().collect(); let sm_a_c: HashSet<usize> = [0].iter().cloned().collect();
        let sm_a_coh = merger.compute_coherence(&sm_a_r, &sm_a_c, &matrix);
        let sm_a_den = merger.compute_density(&sm_a_r, &sm_a_c, &matrix);
        let sm_a_siz = merger.compute_size_metric(&sm_a_r, &sm_a_c);
        let sm_a_sco = merger.score_weights.0 * sm_a_coh + merger.score_weights.1 * sm_a_den + merger.score_weights.2 * sm_a_siz;
        let sm_a = create_test_submatrix(0, &[0], &[0], sm_a_sco, sm_a_coh, sm_a_den, sm_a_siz); // Score 7

        let sm_b_r: HashSet<usize> = [1].iter().cloned().collect(); let sm_b_c: HashSet<usize> = [1].iter().cloned().collect();
        let sm_b_coh = merger.compute_coherence(&sm_b_r, &sm_b_c, &matrix);
        let sm_b_den = merger.compute_density(&sm_b_r, &sm_b_c, &matrix);
        let sm_b_siz = merger.compute_size_metric(&sm_b_r, &sm_b_c);
        let sm_b_sco = merger.score_weights.0 * sm_b_coh + merger.score_weights.1 * sm_b_den + merger.score_weights.2 * sm_b_siz;
        let sm_b = create_test_submatrix(1, &[1], &[1], sm_b_sco, sm_b_coh, sm_b_den, sm_b_siz); // Score 7

        // Merged A+B (AB): rows {0,1}, cols {0,1}. Matrix [[10,10],[10,10]]. Score 7.6

        let sm_c_r: HashSet<usize> = [2].iter().cloned().collect(); let sm_c_c: HashSet<usize> = [2].iter().cloned().collect();
        let sm_c_coh = merger.compute_coherence(&sm_c_r, &sm_c_c, &matrix);
        let sm_c_den = merger.compute_density(&sm_c_r, &sm_c_c, &matrix);
        let sm_c_siz = merger.compute_size_metric(&sm_c_r, &sm_c_c);
        let sm_c_sco = merger.score_weights.0 * sm_c_coh + merger.score_weights.1 * sm_c_den + merger.score_weights.2 * sm_c_siz;
        let sm_c = create_test_submatrix(2, &[2], &[2], sm_c_sco, sm_c_coh, sm_c_den, sm_c_siz); // Score 7

        let sm_d_r: HashSet<usize> = [3].iter().cloned().collect(); let sm_d_c: HashSet<usize> = [3].iter().cloned().collect();
        let sm_d_coh = merger.compute_coherence(&sm_d_r, &sm_d_c, &matrix);
        let sm_d_den = merger.compute_density(&sm_d_r, &sm_d_c, &matrix);
        let sm_d_siz = merger.compute_size_metric(&sm_d_r, &sm_d_c);
        let sm_d_sco = merger.score_weights.0 * sm_d_coh + merger.score_weights.1 * sm_d_den + merger.score_weights.2 * sm_d_siz;
        let sm_d = create_test_submatrix(3, &[3], &[3], sm_d_sco, sm_d_coh, sm_d_den, sm_d_siz); // Score 7
        
        // Merged C+D (CD): rows {2,3}, cols {2,3}. Matrix [[10,10],[10,10]]. Score 7.6

        // Merged (AB) + (CD) : rows {0,1,2,3}, cols {0,1,2,3} but values are block diagonal
        // This matrix is block diagonal. Coherence/Density of the 4x4 block including zeros will be lower.
        // Actual indices for AB are {0,1}x{0,1}. Actual indices for CD are {2,3}x{2,3}.
        // If we merge AB and CD, merged indices are {0,1,2,3} x {0,1,2,3}.
        // Let's test with sm_a, sm_b, sm_c, sm_d as input.
        // Expected: (a,b) merge, (c,d) merge. Then these two merged submatrices do NOT merge because their actual data is far apart,
        // so their combined coherence/density would be low.
        // The overlap between AB and CD is 0.
        
        let local_results = vec![vec![sm_a, sm_b, sm_c, sm_d]];
        let result = merger.merge(local_results, &matrix);

        assert_eq!(result.len(), 2, "Expected 2 submatrices after first level merge, got {}", result.len());
        for sm in &result {
            assert!((sm.score - 7.6).abs() < 1e-9, "Merged submatrix score error. Expected ~7.6, got {}", sm.score);
        }
    }
    
    #[test]
    fn test_hierarchical_merger_empty_input() {
        let merger = HierarchicalMerger::new(0.1, 1.0,1.0,1.0);
        let matrix = Array2::<f64>::zeros((1,1));
        let local_results: Vec<Vec<SubmatrixWithScore>> = vec![];
        let result = merger.merge(local_results, &matrix);
        assert!(result.is_empty(), "Expected empty result for empty input");

        let local_results_inner_empty: Vec<Vec<SubmatrixWithScore>> = vec![vec![]];
        let result_inner_empty = merger.merge(local_results_inner_empty, &matrix);
        assert!(result_inner_empty.is_empty(), "Expected empty result for inner empty input");
    }
} 