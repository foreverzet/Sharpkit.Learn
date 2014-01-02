// -----------------------------------------------------------------------
// <copyright file="ICriterion.cs" company="Sharpkit.Learn">
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
    /// Interface for impurity criteria.
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/_tree.pyx
    /// </remarks>
    internal interface ICriterion
    {
        void init(double[] y,
                  uint y_stride,
                  double[] sample_weight,
                  uint[] samples,
                  uint start,
                  uint end);
        /// <summary>
        /// Reset the criterion at pos=start.
        /// </summary>
        void reset();

        /// <summary>
        /// Update the collected statistics by moving samples[pos:new_pos] from
        ///   the right child to the left child.
        /// </summary>
        void update(uint new_pos);

        /// <summary>
        /// Evaluate the impurity of the current node, i.e. the impurity of
        /// samples[start:end].
        /// </summary>
        /// <returns></returns>
        double node_impurity();

        /// <summary>
        /// Evaluate the impurity in children nodes, i.e. the impurity of
        /// samples[start:pos] + the impurity of samples[pos:end].
        /// </summary>
        double children_impurity();

        /// <summary>
        /// Compute the node value of samples[start:end] into dest.
        /// </summary>
        /// <param name="dest"></param>
        void node_value(double[] dest, uint offset);
    }
}
