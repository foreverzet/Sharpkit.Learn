// -----------------------------------------------------------------------
// <copyright file="RandomSplitter.cs" company="Sharpkit.Learn">
//  Authors: Gilles Louppe <g.louppe@gmail.com>
//           Peter Prettenhofer <peter.prettenhofer@gmail.com>
//           Brian Holt <bdholt1@gmail.com>
//           Noel Dawe <noel@dawe.me>
//           Satrajit Gosh <satrajit.ghosh@gmail.com>
//           Lars Buitinck <L.J.Buitinck@uva.nl>
//           Sergey Zyuzin
//  Licence: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------
namespace Sharpkit.Learn.Tree
{
    using System;

    /// <summary>
    /// Splitter for finding the best random split.
    /// </summary>
    internal class RandomSplitter : SplitterBase
    {
        //Find the best random split on node samples[start:end].
        public RandomSplitter(ICriterion criterion, uint max_features, uint min_samples_leaf, Random random_state) :
            base(criterion, max_features, min_samples_leaf, random_state)
        {
        }

        public override void node_split(ref uint pos, ref uint feature, ref double threshold)
        {
            // Draw random splits and pick the best

            double best_impurity = double.PositiveInfinity;
            uint best_pos = end;
            uint best_feature = 0;
            double best_threshold = 0;

            uint visited_features = 0;

            uint current_feature = 0;
            for (uint f_idx = 0; f_idx < n_features; f_idx++)
            {
                // Draw a feature at random
                uint f_i = n_features - f_idx - 1;
                uint f_j = Util.rand_int(n_features - f_idx, ref rand_r_state);


                uint tmp = features[f_i];
                features[f_i] = features[f_j];
                features[f_j] = tmp;


                current_feature = features[f_i];


                // Find min, max
                double min_feature_value;
                double max_feature_value;
                min_feature_value = max_feature_value = X[X_stride * samples[start] + current_feature];


                for (uint p1 = start + 1; p1 < end; p1++)
                {
                    double current_feature_value = X[X_stride * samples[p1] + current_feature];

                    if (current_feature_value < min_feature_value)
                    {
                        min_feature_value = current_feature_value;
                    }
                    else if (current_feature_value > max_feature_value)
                    {
                        max_feature_value = current_feature_value;
                    }
                }

                if (min_feature_value == max_feature_value)
                {
                    continue;
                }


                // Draw a random threshold
                double current_threshold = (min_feature_value +
                                            Util.rand_double(ref rand_r_state) * (max_feature_value - min_feature_value));

                if (current_threshold == max_feature_value)
                {
                    current_threshold = min_feature_value;
                }

                // Partition
                uint partition_start = start;
                uint partition_end = end;
                uint p = start;

                while (p < partition_end)
                {
                    if (X[X_stride * samples[p] + current_feature] <= current_threshold)
                    {
                        p += 1;
                    }
                    else
                    {
                        partition_end -= 1;

                        tmp = samples[partition_end];
                        samples[partition_end] = samples[p];
                        samples[p] = tmp;
                    }
                }

                uint current_pos = partition_end;

                // Reject if min_samples_leaf is not guaranteed
                if ((((current_pos - start) < min_samples_leaf) ||
                     ((end - current_pos) < min_samples_leaf)))
                {
                    continue;
                }

                // Evaluate split
                this.criterion.Reset();
                this.criterion.Update(current_pos);
                double current_impurity = this.criterion.ChildrenImpurity();


                if (current_impurity < best_impurity)
                {
                    best_impurity = current_impurity;
                    best_pos = current_pos;
                    best_feature = current_feature;
                    best_threshold = current_threshold;
                }


                // Count one more visited feature
                visited_features += 1;


                if (visited_features >= max_features)
                {
                    break;
                }
            }

            // Reorganize into samples[start:best_pos] + samples[best_pos:end]
            if (best_pos < end && current_feature != best_feature)
            {
                uint partition_start = start;
                uint partition_end = end;
                uint p = start;

                while (p < partition_end)
                {
                    if (X[X_stride * samples[p] + best_feature] <= best_threshold)
                    {
                        p += 1;
                    }
                    else
                    {
                        partition_end -= 1;

                        uint tmp = samples[partition_end];
                        samples[partition_end] = samples[p];
                        samples[p] = tmp;
                    }
                }
            }

            // Return values
            pos = best_pos;
            feature = best_feature;
            threshold = best_threshold;
        }
    }
}
