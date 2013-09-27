// -----------------------------------------------------------------------
// <copyright file="Fixes.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Utils
{
    using System;
    using System.Linq;

    /// <summary>
    /// Auxiliary utility methods.
    /// </summary>
    public static class Fixes
    {
        /// <summary>
        /// Returns sorted array of unique items in <paramref name="vals"/>,
        /// populates <paramref name="indices"/> with indices of items in the
        /// returned array which correspond to every item in <paramref name="vals"/>.
        /// </summary>
        /// <typeparam name="T">Type of elements in the array.</typeparam>
        /// <param name="vals">Array of items.</param>
        /// <param name="indices">'Out' array which contains index into returned array for every item in <paramref name="vals"/>.
        /// </param>
        /// <returns>sorted array of unique items in <paramref name="vals"/>.</returns>
        public static T[] Unique<T>(T[] vals, out int[] indices)
        {
            T[] classes = vals.Distinct().OrderBy(v => v).ToArray();
            indices = vals.Select(v => Array.IndexOf(classes, v)).ToArray();
            return classes;
        }
    }
}
