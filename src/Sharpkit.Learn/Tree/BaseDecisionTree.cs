// -----------------------------------------------------------------------
// <copyright file="BaseDecisionTree.cs" company="Sharpkit.Learn">
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
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using Sharpkit.Learn.FeatureSelection;
    using Sharpkit.Learn.Preprocessing;

    /// <summary>
    /// Base class for decision trees.
    /// </summary>
    /// <typeparam name="TLabel">Type of class labels.</typeparam>
    /// <remarks>
    /// Ported from:
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/tree.py
    /// </remarks> 
    public abstract class BaseDecisionTree<TLabel> : ILearntSelector
    {
        private readonly Criterion criterion;
        private readonly Splitter splitter;
        private readonly int? maxDepth;
        private readonly int minSamplesLeaf;
        private readonly MaxFeaturesChoice maxFeatures;
        private readonly Random randomState;
        private int minSamplesSplit;
        private int nOutputs;

        /// <summary>
        /// Initializes a new instance of the BaseDecisionTree class.
        /// </summary>
        /// <param name="criterion">The function to measure the quality of a split.</param>
        /// <param name="splitter">The strategy used to choose the split at each node.</param>
        /// <param name="maxDepth">
        /// The maximum depth of the tree. If <c>null</c>, then nodes are expanded until
        /// all leaves are pure or until all leaves contain less than
        /// <paramref name="minSamplesSplit"/> samples.</param>
        /// <param name="minSamplesSplit">The minimum number of samples required to split an internal node.</param>
        /// <param name="minSamplesLeaf">The minimum number of samples required to be at a leaf node.</param>
        /// <param name="maxFeatures">Number of features to consider when looking for the best split. If null - 
        /// then all features will be considered.</param>
        /// <param name="randomState">random number generator</param>
        public BaseDecisionTree(
            Criterion criterion,
            Splitter splitter,
            int? maxDepth,
            int minSamplesSplit,
            int minSamplesLeaf,
            MaxFeaturesChoice maxFeatures,
            Random randomState)
        {
            this.criterion = criterion;
            this.splitter = splitter;
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.minSamplesLeaf = minSamplesLeaf;
            this.maxFeatures = maxFeatures;
            this.randomState = randomState ?? new Random();
        }

        internal int NFeatures { get; set; }

        /// <summary>
        /// Gets or sets the underlying Tree object.
        /// </summary>
        internal Tree Tree { get; set; }

        /// <summary>
        /// Gets or sets the inferred value of maxFeatures.
        /// </summary>
        internal int MaxFeaturesValue { get; set; }

        /// <summary>
        /// List containing the number of classes for each
        /// output (for multi-output problems).
        /// </summary>
        internal List<uint> NClasses { get; set; }

        internal List<TLabel> ClassesInternal { get; set; }

        /// <summary>
        /// <para>
        /// Return the feature importances.
        /// </para>
        /// <para>
        /// The importance of a feature is computed as the (normalized) total
        /// reduction of the criterion brought by that feature.
        /// It is also known as the Gini importance.
        /// </para>
        /// </summary>
        /// <returns>shape = [n_features]</returns>
        public Vector<double> FeatureImportances()
        {
            if (this.Tree == null)
            {
                throw new InvalidOperationException("Estimator not fitted, call `fit` before `feature_importances_`.");
            }

            return this.Tree.ComputeFeatureImportances();
        }

        /// <summary>
        /// Build a decision tree from the training set (X, y).
        /// </summary>
        /// <param name="x">[n_samples, n_features] The training input samples.</param>
        /// <param name="y">[n_samples, n_outputs] The target values.</param>
        /// <param name="sampleWeight">[n_samples] or None.
        ///  Sample weights. If None, then samples are equally weighted. Splits
        ///  that would create child nodes with net zero or negative weight are
        ///  ignored while searching for a split in each node. In the case of
        ///  classification, splits are also ignored if they would result in any
        ///  single class carrying a negative weight in either child node.</param>
        /// <returns></returns>
        internal void FitRegression(
            Matrix<double> x,
            Matrix<double> y,
            Vector<double> sampleWeight = null)
        {
            // Determine output settings
            int nSamples = x.RowCount;
            this.NFeatures = x.ColumnCount;

            if (y.RowCount != nSamples)
            {
                throw new ArgumentException(
                    string.Format(
                        "Number of labels={0} does not match number of samples={1}",
                        y.RowCount,
                        nSamples));
            }

            // if (y.ndim == 1)
            //    # reshape is necessary to preserve the data contiguity against vs
            //    # [:, np.newaxis] that does not.
            //    y = np.reshape(y, (-1, 1))

            this.nOutputs = y.ColumnCount;
            // this.nOutputs = 1;
            //int[] y_;

            this.ClassesInternal = Enumerable.Repeat(default(TLabel), this.nOutputs).ToList();
            this.NClasses = Enumerable.Repeat(1U, this.nOutputs).ToList();

            // # Check parameters
            FitCommon(x, y, nSamples, sampleWeight, false);
        }

        /// <summary>
        /// Build a decision tree from the training set (X, y).
        /// </summary>
        /// <param name="x">[n_samples, n_features] The training input samples. </param>
        /// <param name="y"> [n_samples] The target values.</param>
        /// <param name="sampleWeight">[n_samples] or None
        /// Sample weights. If None, then samples are equally weighted. Splits
        /// that would create child nodes with net zero or negative weight are
        /// ignored while searching for a split in each node. In the case of
        /// classification, splits are also ignored if they would result in any
        /// single class carrying a negative weight in either child node.</param>
        /// <returns>
        /// Returns this.
        /// </returns>
        internal BaseDecisionTree<TLabel> FitClassification(
            Matrix<double> x,
            TLabel[] y,
            Vector<double> sampleWeight = null)
        {
            // Determine output settings
            int nSamples = x.RowCount;
            this.NFeatures = x.ColumnCount;

            if (y.Length != nSamples)
            {
                throw new ArgumentException(
                    string.Format(
                        "Number of labels={0} does not match number of samples={1}",
                        y.Length,
                        nSamples));
            }

            this.nOutputs = 1;
            int[] y_;

            this.ClassesInternal = new List<TLabel>();
            this.NClasses = new List<uint>();

            var enc = new LabelEncoder<TLabel>();
            y_ = enc.FitTransform(y).ToArray();

            this.ClassesInternal.AddRange(enc.Classes);
            this.NClasses.Add((uint)enc.Classes.Length);

            var yMatrix = y_.Select(v => (double)v).ToArray().ToDenseVector().ToColumnMatrix();
            FitCommon(x, yMatrix, nSamples, sampleWeight, true);

            return this;
        }

        /// <summary>
        /// Predict class value for X.
        /// The predicted class for each sample in X is returned. 
        /// </summary>
        /// <param name="x">[n_samples, n_features] The input samples.</param>
        /// <returns>[n_samples] The predicted classes</returns>
        internal TLabel[] PredictClassification(Matrix<double> x)
        {
            int n_samples = x.RowCount;
            int nFeatures = x.ColumnCount;

            if (this.Tree == null)
            {
                throw new InvalidOperationException("Tree not initialized. Perform a fit first");
            }

            if (this.NFeatures != nFeatures)
            {
                throw new ArgumentException(
                    string.Format(
                        "Number of features of the model must match the input. Model n_features is {0} and input n_features is {1} ",
                        this.NFeatures,
                        nFeatures));
            }

            Matrix<double> proba = this.Tree.predict(x)[0];

            return proba.ArgmaxColumns().Select(v => this.ClassesInternal[v]).ToArray();
        }

        /// <summary>
        /// Predict regression value for X.
        /// The predicted value based on X is returned.
        /// </summary>
        /// <param name="x">[n_samples, n_features] The input samples.</param>
        /// <returns>[n_samples, n_outputs] The predicted classes, or the predict values.</returns>
        internal Matrix<double> PredictRegression(Matrix<double> x)
        {
            int nSamples = x.RowCount;
            int nFeatures = x.ColumnCount;

            CheckFitted();

            if (this.NFeatures != nFeatures)
            {
                throw new ArgumentException(
                    string.Format(
                        "Number of features of the model must match the input. Model n_features is {0} and input n_features is {1} ",
                        this.NFeatures,
                        nFeatures));
            }

            var proba = this.Tree.predict(x);

            var result = DenseMatrix.Create(nSamples, this.nOutputs, (i, j) => 0);
            for (int i = 0; i < proba.Length; i++)
            {
                result.SetColumn(i, proba[i].Column(0));
            }

            return result;
        }

        internal void CheckFitted()
        {
            if (this.Tree == null)
            {
                throw new InvalidOperationException("Tree not initialized. Perform a fit first");
            }
        }

        private void FitCommon(
            Matrix<double> x,
            Matrix<double> y,
            int nSamples,
            Vector<double> sampleWeight,
            bool isClassification)
        {
            int maxDepth = this.maxDepth ?? int.MaxValue;

            if (this.maxFeatures == null)
            {
                this.MaxFeaturesValue = this.NFeatures;
            }
            else
            {
                this.MaxFeaturesValue = maxFeatures.ComputeMaxFeatures(this.NFeatures, isClassification);
            }

            if (this.minSamplesSplit <= 0)
            {
                throw new ArgumentException("min_samples_split must be greater than zero.");
            }

            if (this.minSamplesLeaf <= 0)
            {
                throw new ArgumentException("minSamplesLeaf must be greater than zero.");
            }

            if (maxDepth <= 0)
            {
                throw new ArgumentException("maxDepth must be greater than zero. ");
            }

            if (!(0 < MaxFeaturesValue && MaxFeaturesValue <= this.NFeatures))
            {
                throw new ArgumentException("maxFeatures must be in (0, n_features]");
            }

            if (sampleWeight != null)
            {
                if (sampleWeight.Count != nSamples)
                {
                    throw new ArgumentException(
                        string.Format(
                            "Number of weights={0} does not match number of samples={1}",
                            sampleWeight.Count,
                            nSamples));
                }
            }

            // Set min_samples_split sensibly
            minSamplesSplit = Math.Max(this.minSamplesSplit, 2 * this.minSamplesLeaf);

            // Build tree
            ICriterion criterion = null;
            switch (this.criterion)
            {
                case Criterion.Gini:
                    criterion = new Gini((uint)nOutputs, NClasses.ToArray());
                    break;
                case Criterion.Entropy:
                    criterion = new Entropy((uint)nOutputs, NClasses.ToArray());
                    break;
                case Criterion.Mse:
                    criterion = new MSE((uint)nOutputs);
                    break;
                default:
                    throw new InvalidOperationException("Unknown criterion type");
            }

            SplitterBase splitter = null;
            switch (this.splitter)
            {
                case Splitter.Best:
                    splitter = new BestSplitter(criterion, (uint)this.MaxFeaturesValue, (uint)this.minSamplesLeaf, randomState);
                    break;
                case Splitter.PresortBest:
                    splitter = new PresortBestSplitter(criterion, (uint)this.MaxFeaturesValue, (uint)this.minSamplesLeaf, randomState);
                    break;
                case Splitter.Random:
                    splitter = new RandomSplitter(criterion, (uint)this.MaxFeaturesValue, (uint)this.minSamplesLeaf, randomState);
                    break;
                default:
                    throw new InvalidOperationException("Unknown splitter type");
            }

            this.Tree = new Tree(
                this.NFeatures,
                this.NClasses.ToArray(),
                this.nOutputs,
                splitter,
                (uint)maxDepth,
                (uint)minSamplesSplit,
                (uint)this.minSamplesLeaf);

            this.Tree.build(x, y, sampleWeight: sampleWeight);
        }
    }
}
