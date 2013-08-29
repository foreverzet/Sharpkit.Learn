// -----------------------------------------------------------------------
// <copyright file="LabelBinarizer.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Preprocessing
{
    using MathNet.Numerics.LinearAlgebra.Double;
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Binarize labels in a one-vs-all fashion
    ///
    /// Several regression and binary classification algorithms are
    /// available in the sharpkit. A simple way to extend these algorithms
    /// to the multi-class classification case is to use the so-called
    /// one-vs-all scheme.
    ///
    /// At learning time, this simply consists in learning one regressor
    /// or binary classifier per class. In doing so, one needs to convert
    /// multi-class labels to binary labels (belong or does not belong
    /// to the class). LabelBinarizer makes this process easy with the
    /// transform method.
    ///
    /// At prediction time, one assigns the class for which the corresponding
    /// model gave the greatest confidence. LabelBinarizer makes this easy
    /// with the InverseTransform method.
    /// </summary>
    /// <typeparam name="TLabel"></typeparam>
    /// <example>
    /// var lb = new LabelBinarizer<int>();
    /// lb.fit([1, 2, 6, 4, 2]);
    ///     LabelBinarizer(neg_label=0, pos_label=1)
    /// lb.Classes
    ///     {1, 2, 4, 6}
    /// lb.Transform(new []{1, 6});
    ///   {{1, 0, 0, 0},
    ///    {0, 0, 0, 1}}
    /// </example>
    public class LabelBinarizer<TLabel>  where TLabel:IEquatable<TLabel>
    {
        private readonly int negLabel;
        private readonly int posLabel;

        /// <summary>
        /// Initializes a new instance of the LabelBinarizer class.
        /// </summary>
        /// <param name="negLabel">Value with which negative labels must be encoded.</param>
        /// <param name="posLabel">Value with which positive labels must be encoded.</param>
        public LabelBinarizer(int negLabel=0, int posLabel=1)
        {
            if (negLabel >= posLabel)
                throw new ArgumentException("neg_label must be strictly less than pos_label.");

            this.negLabel = negLabel;
            this.posLabel = posLabel;
        }

        private void CheckFitted()
        {
            if (this.Classes == null)
                throw new InvalidOperationException("LabelBinarizer was not fitted yet.");
        }

        /// <summary>
        /// Holds the label for each class.
        /// </summary>
        public TLabel[] Classes { get; private set; }

        /// <summary>
        /// Fit label binarizer
        /// </summary>
        /// <param name="y">array of shape [n_samples] or sequence of sequences Target values.</param>
        /// <returns>Instance of self.</returns>
        public LabelBinarizer<TLabel> Fit(TLabel[] y)
        {
            this.Classes = y.Distinct().OrderBy(v => v).ToArray();
            return this;
        }

        /// <summary>
        /// Transform multi-class labels to binary labels
        /// The output of transform is sometimes referred to by some authors as the
        /// 1-of-K coding scheme.
        /// </summary>
        /// <param name="y">array of shape [n_samples]. Target values.</param>
        /// <returns>Matrix of shape [n_samples, n_classes]</returns>
        public Matrix Transform(TLabel[] y)
        {
            this.CheckFitted();

            DenseMatrix Y;
            if (this.Classes.Length > 2)
                Y = new DenseMatrix(y.Length, this.Classes.Length);
            else
                Y = new DenseMatrix(y.Length, 1);

            Y.MapInplace(i => i + this.negLabel);

            if (this.Classes.Length > 2)
            {
                // inverse map: label => column index
                Dictionary<TLabel, int> imap = Classes
                    .Select((v, i) => Tuple.Create(i, v))
                    .ToDictionary(t => t.Item2, t => t.Item1);

                for (int i = 0; i < y.Length; i++)
                    Y[i, imap[y[i]]] = this.posLabel;

                return Y;
            }

            else if (this.Classes.Length == 2)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    if (y[i].Equals(this.Classes[1]))
                        Y[i, 0] = this.posLabel;
                }

                return Y;
            }
            else
            {
                // Only one class, returns a matrix with all negative labels.
                return Y;
            }
        }

        /// <summary>
        /// Transform binary labels back to multi-class labels.
        /// </summary>
        /// <param name="y">Array of shape [n_samples]. Target values.</param>
        /// <param name="threshold">Threshold used in the binary and multi-label cases.
        ///
        ///    Use 0 when:
        ///        - Y contains the output of decision_function (classifier)
        ///    Use 0.5 when:
        ///        - Y contains the output of predict_proba
        ///
        ///    If None, the threshold is assumed to be half way between
        ///    neg_label and pos_label.
        /// </param>
        /// <returns>Array of shape [n_samples]. Target values.</returns>
        /// <remarks>
        /// In the case when the binary labels are fractional
        /// (probabilistic), InverseTransform chooses the class with the
        /// greatest value. Typically, this allows to use the output of a
        /// linear model's DecisionFunction method directly as the input
        /// of InverseTransform.
        /// </remarks>
        public TLabel[] InverseTransform(Matrix y, double? threshold=null)
        {
            this.CheckFitted();

            if (threshold == null)
            {
                double half = (this.posLabel - this.negLabel)/2.0;
                threshold = this.negLabel + half;
            }

            var result = new TLabel[y.RowCount];
            if (y.ColumnCount == 1)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = y[i, 0] > threshold ? this.Classes[1] : this.Classes[0];
                }
            }
            else
            {
                foreach (var row in y.RowEnumerator())
                {
                    result[row.Item1] = this.Classes[row.Item2.MaximumIndex()];
                }
            }

            return result;
        }
    }
}
