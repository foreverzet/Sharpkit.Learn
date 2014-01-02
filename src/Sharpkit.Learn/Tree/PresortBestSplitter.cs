// -----------------------------------------------------------------------
// <copyright file="PresortBestSplitter.cs" company="Sharpkit.Learn">
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
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Splitter for finding the best split, using presorting.
    /// </summary>
    internal class PresortBestSplitter : SplitterBase
    {
        private double[] X_old;
        private int[] X_argsorted;
        //cdef INT32_t* X_argsorted_ptr
        private uint X_argsorted_stride;


        private uint n_total_samples;
        private uint[] sample_mask;


        public PresortBestSplitter(ICriterion criterion,
                                   uint max_features,
                                   uint min_samples_leaf,
                                   Random random_state)
            : base(criterion, max_features, min_samples_leaf, random_state)
        {
        }

        public override void init(Matrix<double> X,
                                  Matrix<double> y,
                                  double[] sample_weight)
        {
            //Call parent initializer
            base.init(X, y, sample_weight);


            // Pre-sort X
            if (this.X_old != this.X)
            {
                this.X_old = this.X;
                X_argsorted = X.ArgsortColumns().ToColumnWiseArray();

                this.X_argsorted_stride = (uint)X.RowCount;


                this.n_total_samples = (uint)X.RowCount;
                this.sample_mask = new uint[n_total_samples];
            }
        }

        /// <summary>
        /// Find the best split on node samples[start:end].
        /// </summary>
        /// <param name="pos"></param>
        /// <param name="feature"></param>
        /// <param name="threshold"></param>
        public override void node_split(ref uint pos, ref uint feature, ref double threshold)
        {
            // Find the best split
            double best_impurity = double.PositiveInfinity;
            uint best_pos = end;
            uint best_feature = 0;
            double best_threshold = 0;


            uint current_feature = 0;

            uint visited_features = 0;

            // Set sample mask
            for (uint p1 = start; p1 < end; p1++)
            {
                sample_mask[samples[p1]] = 1;
            }


            // Look for splits
            for (uint f_idx = 0; f_idx < n_features; f_idx++)
            {
                // Draw a feature at random
                uint f_i = n_features - f_idx - 1;
                uint f_j = Util.rand_int(n_features - f_idx, ref rand_r_state);


                uint tmp = features[f_i];
                features[f_i] = features[f_j];
                features[f_j] = tmp;

                current_feature = features[f_i];


                // Extract ordering from X_argsorted
                uint p = start;

                for (int i = 0; i < n_total_samples; i++)
                {
                    uint j = (uint)X_argsorted[X_argsorted_stride * current_feature + i];
                    if (sample_mask[j] == 1)
                    {
                        samples[p] = j;
                        p += 1;
                    }
                }

                // Evaluate all splits
                this.criterion.reset();
                p = start;
                while (p < end)
                {
                    while (((p + 1 < end) &&
                            (X[X_stride * samples[p + 1] + current_feature] <=
                             X[X_stride * samples[p] + current_feature] + 1e-7)))
                    {
                        p += 1;
                    }


                    // (p + 1 >= end) or (X[samples[p + 1], current_feature] >
                    //                   X[samples[p], current_feature])
                    p += 1;
                    // (p >= end) or (X[samples[p], current_feature] >
                    //                X[samples[p - 1], current_feature])


                    if (p < end)
                    {
                        uint current_pos = p;


                        // Reject if min_samples_leaf is not guaranteed
                        if ((((current_pos - start) < min_samples_leaf) ||
                             ((end - current_pos) < min_samples_leaf)))
                        {
                            continue;
                        }


                        this.criterion.update(current_pos);
                        double current_impurity = this.criterion.children_impurity();


                        if (current_impurity < (best_impurity - 1e-7))
                        {
                            best_impurity = current_impurity;
                            best_pos = current_pos;
                            best_feature = current_feature;


                            double current_threshold = (X[X_stride * samples[p - 1] + current_feature] +
                                                    X[X_stride * samples[p] + current_feature]) / 2.0;


                            if (current_threshold == X[X_stride * samples[p] + current_feature])
                            {
                                current_threshold = X[X_stride * samples[p - 1] + current_feature];
                            }


                            best_threshold = current_threshold;
                        }
                    }
                }

                if (best_pos == end) // No valid split was ever found
                {
                    continue;
                }

                // Count one more visited feature
                visited_features += 1;

                if (visited_features >= max_features)
                {
                    break;
                }
            }

            // Reorganize into samples[start:best_pos] + samples[best_pos:end]
            if (best_pos < end)
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

            // Reset sample mask
            for (uint p1 = start; p1 < end; p1++)
            {
                sample_mask[samples[p1]] = 0;
            }


            // Return values
            pos = best_pos;
            feature = best_feature;
            threshold = best_threshold;
        }
    }
}