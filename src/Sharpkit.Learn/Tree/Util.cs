// -----------------------------------------------------------------------
// <copyright file="Util.cs" company="Sharpkit.Learn">
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
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/_tree.pyx
    /// </remarks>
    internal static class Util
    {
        public const uint RAND_R_MAX = 0x7FFFFFFF;

        /// <summary>
        /// Generate a random integer in [0; end).
        /// </summary>
        /// <param name="end"></param>
        /// <param name="random_state"></param>
        /// <returns></returns>
        public static uint rand_int(uint end, ref uint random_state)
        {
            return our_rand_r(ref random_state) % end;
        }



        // rand_r replacement using a 32bit XorShift generator
        // See http://www.jstatsoft.org/v08/i14/paper for details
        public static uint our_rand_r(ref uint seed)
        {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;

            return seed % (RAND_R_MAX + 1);
        }

        /// <summary>
        /// Generate a random double in [0; 1).
        /// </summary>
        /// <param name="random_state"></param>
        /// <returns></returns>
        public static double rand_double(ref uint random_state)
        {
            return (double)our_rand_r(ref random_state) / RAND_R_MAX;
        }

        public static double log(double x)
        {
            return Math.Log(x) / Math.Log(2.0);
        }
    }
}
