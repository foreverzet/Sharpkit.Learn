// -----------------------------------------------------------------------
// <copyright file="Gini.cs" company="Sharpkit.Learn">
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
    /// Gini Index impurity criteria.
    ///
    /// Let the target be a classification outcome taking values in 0, 1, ..., K-1.
    /// If node m represents a region Rm with Nm observations, then let
    /// pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)
    /// be the proportion of class k observations in node m.
    ///
    /// The Gini Index is then defined as:
    ///
    ///
    ///    index = \sum_{k=0}^{K-1} pmk (1 - pmk)
    ///         = 1 - \sum_{k=0}^{K-1} pmk ** 2
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/_tree.pyx
    /// </remarks>
    internal class Gini : ClassificationCriterion
    {
        public Gini(uint n_outputs, uint[] n_classes) : base(n_outputs, n_classes)
        {
        }

        /// <summary>
        /// Evaluate the impurity of the current node, i.e. the impurity of
        ///   samples[start:end].
        /// </summary>
        /// <returns></returns>
        public override double NodeImpurity()
        {
            uint label_count_total_offset = 0;

            double total = 0.0;

            for (int k = 0; k < n_outputs; k++)
            {
                double gini = 0.0;


                for (int c = 0; c < n_classes[k]; c++)
                {
                    double tmp = label_count_total[c + label_count_total_offset];
                    gini += tmp*tmp;
                }

                gini = 1.0 - gini/(weighted_n_node_samples*
                                   weighted_n_node_samples);


                total += gini;
                label_count_total_offset += label_count_stride;
            }

            return total/n_outputs;
        }

        /// <summary>
        /// Evaluate the impurity in children nodes, i.e. the impurity of
        /// samples[start:pos] + the impurity of samples[pos:end].
        /// </summary>
        /// <returns></returns>
        public override double ChildrenImpurity()
        {
            uint label_count_left_offset = 0;
            uint label_count_right_offset = 0;

            double total = 0.0;

            for (int k = 0; k < n_outputs; k++)
            {
                double gini_left = 0.0;
                double gini_right = 0.0;


                for (int c = 0; c < n_classes[k]; c++)
                {
                    double tmp = label_count_left[c + label_count_left_offset];
                    gini_left += tmp*tmp;
                    tmp = label_count_right[c + label_count_right_offset];
                    gini_right += tmp*tmp;
                }

                gini_left = 1.0 - gini_left/(weighted_n_left*
                                             weighted_n_left);
                gini_right = 1.0 - gini_right/(weighted_n_right*
                                               weighted_n_right);


                total += weighted_n_left*gini_left;
                total += weighted_n_right*gini_right;
                label_count_left_offset += label_count_stride;
                label_count_right_offset += label_count_stride;
            }

            return total/(weighted_n_node_samples*n_outputs);
        }
    }
}
