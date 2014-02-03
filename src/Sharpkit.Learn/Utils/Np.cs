// -----------------------------------------------------------------------
// <copyright file="Np.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Utils
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public static class Np
    {
        public static int[] BinCount(int[] vals, int? minLength = null)
        {
            var histogram = new Dictionary<int, int>();
            foreach (var ind in vals)
            {
                int val;
                histogram.TryGetValue(ind, out val);
                val++;
                histogram[ind] = val;
            }

            int[] result = new int[Math.Max(minLength ?? -1, vals.Max() + 1)];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = histogram.ContainsKey(i) ? histogram[i] : 0;
            }

            return result;
        }

        public static int[] Permutation(this Random random, int n)
        {
            List<int> result = new List<int>();
            var items = Enumerable.Range(0, n).ToList();
            while (items.Count > 0)
            {
                var index = random.Next(items.Count);
                result.Add(items[index]);
                items.RemoveAt(index);
            }

            return result.ToArray();
        }
        
    }
}
