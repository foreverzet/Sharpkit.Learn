// -----------------------------------------------------------------------
// <copyright file="Entropy.cs" company="Sharpkit.Learn">
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
    /// Cross Entropy impurity criteria.
    ///
    /// Let the target be a classification outcome taking values in 0, 1, ..., K-1.
    /// If node m represents a region Rm with Nm observations, then let
    /// pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)
    ///
    /// be the proportion of class k observations in node m.
    ///
    /// The cross-entropy is then defined as
    ///
    /// cross-entropy = - \sum_{k=0}^{K-1} pmk log(pmk)
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/_tree.pyx
    /// </remarks>
    internal class Entropy : ClassificationCriterion
    {
        public Entropy(uint n_outputs, uint[] n_classes) : base(n_outputs, n_classes)
        {
        }

        /// <summary>
        /// Evaluate the impurity of the current node, i.e. the impurity of
        ///samples[start:end].
        /// </summary>
        /// <returns></returns>
        public override double NodeImpurity()
        {
            double total = 0.0;
            uint label_count_total_offset = 0;
            for (int k = 0; k < n_outputs; k++)
            {
                double entropy = 0.0;

                for (int c = 0; c < n_classes[k]; c++)
                {
                    double tmp = label_count_total[label_count_total_offset + c];
                    if (tmp > 0.0)
                    {
                        tmp /= weighted_n_node_samples;
                        entropy -= tmp*Math.Log(tmp);
                    }
                }

                total += entropy;
                label_count_total_offset += label_count_stride;
            }

            return total/n_outputs;
        }

        /// <summary>
        /// Evaluate the impurity in children nodes, i.e. the impurity of
        ///samples[start:pos] + the impurity of samples[pos:end].
        /// </summary>
        /// <returns></returns>
        public override double ChildrenImpurity()
        {
            double total = 0.0;
            uint label_count_left_offset = 0;
            uint label_count_right_offset = 0;


            for (int k = 0; k < n_outputs; k++)
            {
                double entropy_left = 0.0;
                double entropy_right = 0.0;


                for (int c = 0; c < n_classes[k]; c++)
                {
                    double tmp = label_count_left[label_count_left_offset + c];
                    if (tmp > 0.0)
                    {
                        tmp /= weighted_n_left;
                        entropy_left -= tmp*Math.Log(tmp);
                    }

                    tmp = label_count_right[label_count_right_offset + c];
                    if (tmp > 0.0)
                    {
                        tmp /= weighted_n_right;
                        entropy_right -= tmp*Math.Log(tmp);
                    }
                }

                total += weighted_n_left*entropy_left;
                total += weighted_n_right*entropy_right;
                label_count_left_offset += label_count_stride;
                label_count_right_offset += label_count_stride;
            }

            return total/(weighted_n_node_samples*n_outputs);
        }
    }
}
