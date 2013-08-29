// -----------------------------------------------------------------------
// <copyright file="LabelEncoder.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Preprocessing
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Encode labels with value between 0 and n_classes-1.
    /// </summary>
    /// <example>
    /// `LabelEncoder` can be used to normalize labels.
    ///
    /// var le = new.LabelEncoder();
    /// le.Fit(new [] {1, 2, 2, 6});
    /// le.Classes
    ///     {1, 2, 6}
    /// le.Transform(new[] {1, 1, 2, 6})
    ///     {0, 0, 1, 2}
    /// le.InverseTransform(new[] {0, 0, 1, 2})
    ///     {1, 1, 2, 6}
    ///
    /// It can also be used to transform non-numerical labels to numerical labels.
    /// 
    /// var le = new LabelEncoder();
    /// le.Fit(new[] {"paris", "paris", "tokyo", "amsterdam"});
    /// le.Classes
    ///     {"amsterdam', "paris", "tokyo"}
    /// le.Transform(new[] {"tokyo", "tokyo", "paris"})
    ///     {2, 2, 1}
    /// le.InverseTransform(new[] {2, 2, 1});
    ///     {"tokyo", "tokyo", "paris"}
    /// </example>
    public class LabelEncoder<TLabel>
    {
        private void CheckFitted()
        {
            if (this.Classes == null)
            {
                throw new InvalidOperationException("LabelNormalizer was not fitted yet.");
            }
        }

        /// <summary>
        /// Fit label encoder.
        /// </summary>
        /// <param name="y">Target values. [n_samples]</param>
        /// <returns>Returns an instance of self.</returns>
        public LabelEncoder<TLabel> Fit(TLabel[] y)
        {
            this.Classes = y.Distinct().OrderBy(a => a).ToArray();
            return this;
        }

        /// <summary>
        /// Holds the label for each class.
        /// </summary>
        public TLabel[] Classes { get; private set; }

        /// <summary>
        /// Fit label encoder and return encoded labels.
        /// </summary>
        /// <param name="y">Target values. [n_samples]</param>
        /// <returns>Array [n_samples].</returns>
        public int[] FitTransform(TLabel[] y)
        {
            return Fit(y).Transform(y);
        }

        /// <summary>
        /// Transform labels to normalized encoding.
        /// </summary>
        /// <param name="y">Target values. [n_samples]</param>
        /// <returns>Array [n_samples].</returns>
        public int[] Transform(TLabel[] y)
        {
            CheckFitted();

            var classes = y.Distinct().ToList();
            var diff = new HashSet<TLabel>(classes);
            diff.ExceptWith(new HashSet<TLabel>(this.Classes));
            if (diff.Any())
            {
                throw new ArgumentException("y contains new labels: {0}", string.Concat(",", diff));
            }

            var dict = this.Classes.Select((label, index) => Tuple.Create(label, index)).ToDictionary(t => t.Item1,
                                                                                                      t => t.Item2);
            return y.Select(l => dict[l]).ToArray();
        }

        /// <summary>
        /// Transform labels back to original encoding.
        /// </summary>
        /// <param name="y">Target values. [n_samples]</param>
        /// <returns>Array. [n_samples]</returns>
        public TLabel[] InverseTransform(int[] y)
        {
            CheckFitted();

            return y.Select(a => this.Classes[a]).ToArray();
        }
    }
}
