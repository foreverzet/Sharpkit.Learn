// -----------------------------------------------------------------------
// <copyright file="MSE.cs" company="Sharpkit.Learn">
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
    /// Mean squared error impurity criterion.
    /// 
    /// MSE = var_left + var_right
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/_tree.pyx
    /// </remarks>
    internal class MSE : RegressionCriterion
    {
        public MSE(uint n_outputs) : base(n_outputs)
        {
        }

        /// <summary>
        /// Evaluate the impurity of the current node, i.e. the impurity of
        ///   samples[start:end].
        /// </summary>
        /// <returns></returns>
        public override double node_impurity()
        {
            double total = 0.0;


            for (int k = 0; k < n_outputs; k++)
            {
                total += (sq_sum_total[k] -
                          weighted_n_node_samples*(mean_total[k]*
                                                   mean_total[k]));
            }

            return total/n_outputs;
        }

        /// <summary>
        /// Evaluate the impurity in children nodes, i.e. the impurity of
        /// samples[start:pos] + the impurity of samples[pos:end].
        /// </summary>
        /// <returns></returns>
        public override double children_impurity()
        {
            double total = 0.0;

            for (int k = 0; k < n_outputs; k++)
            {
                total += var_left[k];
                total += var_right[k];
            }

            return total/n_outputs;
        }
    }
}
