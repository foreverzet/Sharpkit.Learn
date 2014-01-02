// -----------------------------------------------------------------------
// <copyright file="Criterion.cs" company="Sharpkit.Learn">
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
    /// Impurity criteria.
    /// </summary>
    public enum Criterion
    {
        /// <summary>
        /// Gini Index impurity criteria.
        /// </summary>
        Gini,

        /// <summary>
        /// Cross Entropy impurity criteria.
        /// </summary>
        Entropy,

        /// <summary>
        /// Mean squared error impurity criterion.
        /// </summary>
        Mse
    }
}
