// -----------------------------------------------------------------------
// <copyright file="LabelBinarizer.cs" company="Sharpkit.Learn">
// Authors: Alexandre Gramfort &lt;alexandre.gramfort@inria.fr>
//         Mathieu Blondel &lt;mathieu@mblondel.org>
//         Olivier Grisel &lt;olivier.grisel@ensta.org>
//         Andreas Mueller &lt;amueller@ais.uni-bonn.de>
// License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

using MathNet.Numerics.LinearAlgebra.Double;

namespace Sharpkit.Learn.Preprocessing
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Binarize labels in a one-vs-all fashion
    /// <para>
    /// Several regression and binary classification algorithms are
    /// available in the sharpkit. A simple way to extend these algorithms
    /// to the multi-class classification case is to use the so-called
    /// one-vs-all scheme.
    /// </para>
    /// <para>
    /// At learning time, this simply consists in learning one regressor
    /// or binary classifier per class. In doing so, one needs to convert
    /// multi-class labels to binary labels (belong or does not belong
    /// to the class). LabelBinarizer makes this process easy with the
    /// transform method.
    /// </para>
    /// <para>
    /// At prediction time, one assigns the class for which the corresponding
    /// model gave the greatest confidence. LabelBinarizer makes this easy
    /// with the InverseTransform method.
    /// </para>
    /// </summary>
    /// <typeparam name="TLabel">Type of class label.</typeparam>
    /// <example>
    /// var lb = new LabelBinarizer&lt;int>();
    /// lb.fit([1, 2, 6, 4, 2]);
    ///     LabelBinarizer(neg_label=0, pos_label=1)
    /// lb.Classes
    ///     {1, 2, 4, 6}
    /// lb.Transform(new []{1, 6});
    ///   {{1, 0, 0, 0},
    ///    {0, 0, 0, 1}}
    /// </example>
    public class LabelBinarizer<TLabel> where TLabel : IEquatable<TLabel>
    {
        /// <summary>
        /// Value to use as negative label.
        /// </summary>
        private readonly int negLabel;

        /// <summary>
        /// Value to use as positive label.
        /// </summary>
        private readonly int posLabel;

        /// <summary>
        /// Initializes a new instance of the LabelBinarizer class.
        /// </summary>
        /// <param name="negLabel">Value with which negative labels must be encoded.</param>
        /// <param name="posLabel">Value with which positive labels must be encoded.</param>
        public LabelBinarizer(int negLabel = 0, int posLabel = 1)
        {
            if (negLabel >= posLabel)
            {
                throw new ArgumentException("negLabel must be strictly less than posLabel.");
            }

            this.negLabel = negLabel;
            this.posLabel = posLabel;
        }

        /// <summary>
        /// Gets label for each class.
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
        public Matrix<double> Transform(TLabel[] y)
        {
            this.CheckFitted();

            DenseMatrix yMatrix;
            if (this.Classes.Length > 2)
            {
                yMatrix = new DenseMatrix(y.Length, this.Classes.Length);
            }
            else
            {
                yMatrix = new DenseMatrix(y.Length, 1);
            }

            yMatrix.MapInplace(i => i + this.negLabel);

            if (this.Classes.Length > 2)
            {
                // inverse map: label => column index
                Dictionary<TLabel, int> imap = Classes
                    .Select((v, i) => Tuple.Create(i, v))
                    .ToDictionary(t => t.Item2, t => t.Item1);

                for (int i = 0; i < y.Length; i++)
                {
                    yMatrix[i, imap[y[i]]] = this.posLabel;
                }

                return yMatrix;
            }
            else if (this.Classes.Length == 2)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    if (y[i].Equals(this.Classes[1]))
                    {
                        yMatrix[i, 0] = this.posLabel;
                    }
                }

                return yMatrix;
            }
            else
            {
                // Only one class, returns a matrix with all negative labels.
                return yMatrix;
            }
        }

        /// <summary>
        /// Transform binary labels back to multi-class labels.
        /// </summary>
        /// <param name="y">Array of shape [nSamples]. Target values.</param>
        /// <param name="threshold">Threshold used in the binary and multi-label cases.
        ///    Use 0 when:
        ///        - Y contains the output of decision_function (classifier)
        ///    Use 0.5 when:
        ///        - Y contains the output of predict_proba
        ///    If None, the threshold is assumed to be half way between
        ///    negLabel and posLabel.
        /// </param>
        /// <returns>Array of shape [n_samples]. Target values.</returns>
        /// <remarks>
        /// In the case when the binary labels are fractional
        /// (probabilistic), InverseTransform chooses the class with the
        /// greatest value. Typically, this allows to use the output of a
        /// linear model's DecisionFunction method directly as the input
        /// of InverseTransform.
        /// </remarks>
        public TLabel[] InverseTransform(Matrix<double> y, double? threshold = null)
        {
            this.CheckFitted();

            if (threshold == null)
            {
                double half = (this.posLabel - this.negLabel) / 2.0;
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

        /// <summary>
        /// Ensures that binarizer is fitted.
        /// </summary>
        private void CheckFitted()
        {
            if (this.Classes == null)
            {
                throw new InvalidOperationException("LabelBinarizer was not fitted yet.");
            }
        }
    }
}
