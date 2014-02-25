// -----------------------------------------------------------------------
// <copyright file="BestSplitter.cs" company="Sharpkit.Learn">
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
    /// Splitter for finding the best split.
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/_tree.pyx
    /// </remarks>
    internal class BestSplitter : SplitterBase
    {
        public BestSplitter(ICriterion criterion, uint max_features, uint min_samples_leaf, Random random_state)
            : base(criterion, max_features, min_samples_leaf, random_state)
        {
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
            int visited_features = 0;

            for (uint f_idx = 0; f_idx < n_features; f_idx++)
            {
                // Draw a feature at random
                uint f_i = n_features - f_idx - 1;
                uint f_j = Util.rand_int(n_features - f_idx, ref rand_r_state);

                uint tmp = features[f_i];
                features[f_i] = features[f_j];
                features[f_j] = tmp;

                uint current_feature = features[f_i];

                // Sort samples along that feature
                Sort(X, X_stride, current_feature, samples, start, end - start);

                // Evaluate all splits
                this.criterion.Reset();
                uint p = start;

                while (p < end)
                {
                    while ((p + 1 < end) &&
                           (X[X_stride * samples[p + 1] + current_feature] <=
                            X[X_stride * samples[p] + current_feature] + 1e-7))
                    {
                        p += 1;
                    }

                    // (p + 1 >= end) or (X[samples[p + 1], current_feature] >
                    //                    X[samples[p], current_feature])
                    p += 1;
                    // (p >= end) or (X[samples[p], current_feature] >
                    //                X[samples[p - 1], current_feature])


                    if (p < end)
                    {
                        uint current_pos = p;

                        // Reject if min_samples_leaf is not guaranteed
                        if (((current_pos - start) < min_samples_leaf) ||
                            ((end - current_pos) < min_samples_leaf))
                        {
                            continue;
                        }


                        this.criterion.Update(current_pos);
                        double current_impurity = this.criterion.ChildrenImpurity();

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
                        p += 1;
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

        /// <summary>
        /// In-place sorting of samples[start:end] using
        ///  X[sample[i], current_feature] as key.
        /// </summary>
        private static void Sort(
            double[] x,
            uint xStride,
            uint currentFeature,
            uint[] samples,
            uint samplesOffset,
            uint length)
        {
            //# Heapsort, adapted from Numerical Recipes in C
            uint n = length;
            uint parent = length / 2;

            while (true)
            {
                uint tmp;
                if (parent > 0)
                {
                    parent -= 1;
                    tmp = samples[parent + samplesOffset];
                }
                else
                {
                    n -= 1;
                    if (n == 0)
                    {
                        return;
                    }
                    tmp = samples[n + samplesOffset];
                    samples[n + samplesOffset] = samples[0 + samplesOffset];
                }

                double tmp_value = x[xStride * tmp + currentFeature];
                uint index = parent;
                uint child = index * 2 + 1;

                while (child < n)
                {
                    if ((child + 1 < n) &&
                        (x[xStride * samples[child + 1 + samplesOffset] + currentFeature] >
                         x[xStride * samples[child + samplesOffset] + currentFeature]))
                    {
                        child += 1;
                    }


                    if (x[xStride * samples[child + samplesOffset] + currentFeature] > tmp_value)
                    {
                        samples[index + samplesOffset] = samples[child + samplesOffset];
                        index = child;
                        child = index * 2 + 1;
                    }
                    else
                    {
                        break;
                    }
                }
                samples[index + samplesOffset] = tmp;
            }
        }
    }
}
