// -----------------------------------------------------------------------
// <copyright file="Splitter.cs" company="Sharpkit.Learn">
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
    /// TODO: Update summary.
    /// </summary>
    public enum Splitter
    {
        /// <summary>
        /// Splitter for finding the best split.
        /// </summary>
        Best,

        /// <summary>
        /// Splitter for finding the best split, using presorting.
        /// </summary>
        PresortBest,

        /// <summary>
        /// 
        /// </summary>
        Random
    }
}
