// -----------------------------------------------------------------------
// <copyright file="MaxFeaturesChoice.cs" company="Sharpkit.Learn">
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
    /// Strategy for determining number of features to consider when looking for the best split.
    /// </summary>
    public class MaxFeaturesChoice : IEquatable<MaxFeaturesChoice>
    {
        private Func<int, bool, int> f;
        private readonly string name;
        private readonly int value;

        /// <summary>
        /// Creates new <see cref="MaxFeaturesChoice"/> which considers
        /// sqrt(n_features) features at each split.
        /// </summary>
        /// <returns>Instance of <see cref="MaxFeaturesChoice"/>.</returns>
        public static MaxFeaturesChoice Auto()
        {
            return new MaxFeaturesChoice((nFeatures, isClassification)=>
                                             {
                                                 if (isClassification)
                                                     return Math.Max(1, (int)Math.Sqrt(nFeatures));
                                                 else
                                                     return nFeatures;
                                             }, "Auto");
        }

        /// <summary>
        /// Creates new <see cref="MaxFeaturesChoice"/> which considers
        /// sqrt(n_features) features at each split.
        /// </summary>
        /// <returns>Instance of <see cref="MaxFeaturesChoice"/>.</returns>
        public static MaxFeaturesChoice Sqrt()
        {
            return new MaxFeaturesChoice((nFeatures, nClassification) => Math.Max(1, (int)Math.Sqrt(nFeatures)), "Sqrt");
        }

        /// <summary>
        /// Creates new <see cref="MaxFeaturesChoice"/> which considers
        /// log2(n_features) features at each split.
        /// </summary>
        /// <returns>Instance of <see cref="MaxFeaturesChoice"/>.</returns>
        public static MaxFeaturesChoice Log2()
        {
            return new MaxFeaturesChoice((nFeatures, nClassification) => Math.Max(1, (int)Math.Log(nFeatures, 2)), "Log2");
        }

        /// <summary>
        /// Creates new <see cref="MaxFeaturesChoice"/> which considers
        /// n_features * <paramref name="fraction"/> features at each split.
        /// </summary>
        /// <param name="fraction">Fraction of features to consider.</param>
        /// <returns>Instance of <see cref="MaxFeaturesChoice"/>.</returns>
        public static MaxFeaturesChoice Fraction(double fraction)
        {
            return new MaxFeaturesChoice((nFeatures, isClassification)=> (int)(nFeatures* fraction), "Fraction");
        }

        /// <summary>
        /// Creates new <see cref="MaxFeaturesChoice"/> which considers
        /// <paramref name="maxFeatures"/> features at each split.
        /// </summary>
        /// <param name="maxFeatures">Number of features to consider at each split.</param>
        /// <returns>Instance of <see cref="MaxFeaturesChoice"/>.</returns>
        public static MaxFeaturesChoice Value(int maxFeatures)
        {
            return new MaxFeaturesChoice(maxFeatures);
        }

        private MaxFeaturesChoice(Func<int, bool,  int> f, string name)
        {
            this.f = f;
            this.name = name;
        }

        private MaxFeaturesChoice(int maxFeatures)
        {
            this.value = maxFeatures;
        }

        public bool Equals(MaxFeaturesChoice other)
        {
            return this.name == other.name && this.value == other.value;
        }

        public override string ToString()
        {
            if (name != null) return name;
            return value.ToString();
        }

        internal int ComputeMaxFeatures(int n_features_, bool isClassification)
        {
            if (f != null)
            {
                return f(n_features_, isClassification);
            }
            else
            {
                return value;
            }
        }

    }
}