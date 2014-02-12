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

using Sharpkit.Learn.FeatureSelection;

namespace Sharpkit.Learn.Tree
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using MathNet.Numerics.LinearAlgebra.Double;
    using Sharpkit.Learn.Preprocessing;

    /// <summary>
    /// Base class for decision trees.
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/tree.py
    /// </remarks> 
    public abstract class BaseDecisionTree<TLabel> : ILearntSelector
    {
        private readonly Criterion criterion;
        private readonly Splitter splitter;
        private readonly int? maxDepth;
        private int minSamplesSplit;
        private readonly int minSamplesLeaf;
        private readonly MaxFeaturesChoice maxFeatures;
        private readonly Random randomState;
        internal int nFeatures;
        
        /// <summary>
        /// The underlying Tree object.
        /// </summary>
        internal Tree tree_;
        private int nOutputs;

        /// <summary>
        /// The inferred value of maxFeatures.
        /// </summary>
        internal int maxFeaturesValue;
        internal List<uint> nClasses;
        internal List<TLabel> classes;

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
        internal BaseDecisionTree<TLabel> fitRegression(
            Matrix<double> x,
            Matrix<double> y,
            Vector<double> sampleWeight = null)
        {
            // Determine output settings
            int nSamples = x.RowCount;
            this.nFeatures = x.ColumnCount;


            if (y.RowCount != nSamples)
            {
                throw new ArgumentException(string.Format("Number of labels={0} does not match number of samples={1}",
                                                          y.RowCount,
                                                          nSamples));
            }

            //if (y.ndim == 1)
            //    # reshape is necessary to preserve the data contiguity against vs
            //    # [:, np.newaxis] that does not.
            //    y = np.reshape(y, (-1, 1))


            this.nOutputs = y.ColumnCount;
            //this.nOutputs = 1;
            int[] y_;

            this.classes = Enumerable.Repeat(default(TLabel), this.nOutputs).ToList();
            this.nClasses = Enumerable.Repeat(1U, this.nOutputs).ToList();


            //# Check parameters
            fitCommon(x, y, nSamples, sampleWeight, false);


            return this;
        }

        private void fitCommon(
            Matrix<double> x,
            Matrix<double> y,
            int nSamples,
            Vector<double> sampleWeight,
            bool isClassification)
        {
            int maxDepth = this.maxDepth ?? int.MaxValue;

            if (this.maxFeatures == null)
            {
                this.maxFeaturesValue = this.nFeatures;
            }
            else
            {
                this.maxFeaturesValue = maxFeatures.ComputeMaxFeatures(this.nFeatures, isClassification);
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

            if (!(0 < maxFeaturesValue && maxFeaturesValue <= this.nFeatures))
            {
                throw new ArgumentException("maxFeatures must be in (0, n_features]");
            }

            if (sampleWeight != null)
            {
                if (sampleWeight.Count != nSamples)
                {
                    throw new ArgumentException(
                        string.Format("Number of weights={0} does not match number of samples={1}",
                                      sampleWeight.Count, nSamples));
                }
            }


            // Set min_samples_split sensibly
            minSamplesSplit = Math.Max(this.minSamplesSplit,
                                         2*this.minSamplesLeaf);


            // Build tree
            ICriterion criterion = null;
            switch (this.criterion)
            {
                case Criterion.Gini:
                    criterion = new Gini((uint)nOutputs, nClasses.ToArray());
                    break;
                case Criterion.Entropy:
                    criterion = new Entropy((uint)nOutputs, nClasses.ToArray());
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
                    splitter = new BestSplitter(criterion, (uint)this.maxFeaturesValue, (uint)this.minSamplesLeaf, randomState);
                    break;
                case Splitter.PresortBest:
                    splitter = new PresortBestSplitter(criterion, (uint)this.maxFeaturesValue, (uint)this.minSamplesLeaf, randomState);
                    break;
                case Splitter.Random:
                    splitter = new RandomSplitter(criterion, (uint)this.maxFeaturesValue, (uint)this.minSamplesLeaf, randomState);
                    break;
                default:
                    throw new InvalidOperationException("Unknown splitter type");
            }

            this.tree_ = new Tree(this.nFeatures, this.nClasses.ToArray(),
                                  this.nOutputs, splitter, (uint)maxDepth,
                                  (uint)minSamplesSplit, (uint)this.minSamplesLeaf);


            this.tree_.build(x, y, sampleWeight: sampleWeight);
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
        internal BaseDecisionTree<TLabel> fitClassification(
            Matrix<double> x,
            TLabel[] y,
            Vector<double> sampleWeight = null)
        {
            // Determine output settings
            int nSamples = x.RowCount;
            this.nFeatures = x.ColumnCount;

            if (y.Length != nSamples)
            {
                throw new ArgumentException(string.Format("Number of labels={0} does not match number of samples={1}",
                                                          y.Length,
                                                          nSamples));
            }

            this.nOutputs = 1;
            int[] y_;


            this.classes = new List<TLabel>();
            this.nClasses = new List<uint>();

            var enc = new LabelEncoder<TLabel>();
            y_ = enc.FitTransform(y).ToArray();

            this.classes.AddRange(enc.Classes);
            this.nClasses.Add((uint)enc.Classes.Length);

            var yMatrix = y_.Select(v => (double)v).ToArray().ToDenseVector().ToColumnMatrix();
            fitCommon(x, yMatrix, nSamples, sampleWeight, true);


            return this;
        }

        /// <summary>
        /// Predict class value for X.
        /// The predicted class for each sample in X is returned. 
        /// </summary>
        /// <param name="X">[n_samples, n_features] The input samples.</param>
        /// <returns>[n_samples] The predicted classes</returns>
        internal TLabel[] predictClassification(Matrix<double> X)
        {
            int n_samples = X.RowCount;
            int n_features = X.ColumnCount;


            if (this.tree_ == null)
            {
                throw new InvalidOperationException("Tree not initialized. Perform a fit first");
            }


            if (this.nFeatures != n_features)
            {
                throw new ArgumentException(
                    String.Format(
                        "Number of features of the model must match the input. Model n_features is {0} and input n_features is {1} ",
                        this.nFeatures, n_features));
            }


            Matrix<double> proba = this.tree_.predict(X)[0];

            return proba.ArgmaxColumns().Select(v => this.classes[v]).ToArray();
        }

        /// <summary>
        /// Predict regression value for X.
        /// The predicted value based on X is returned.
        /// </summary>
        /// <param name="X">[n_samples, n_features] The input samples.</param>
        /// <returns>[n_samples, n_outputs] The predicted classes, or the predict values.</returns>
        internal Matrix<double> predictRegression(Matrix<double> X)
        {
            int n_samples = X.RowCount;
            int n_features = X.ColumnCount;


            CheckFitted();


            if (this.nFeatures != n_features)
            {
                throw new ArgumentException(
                    String.Format(
                        "Number of features of the model must match the input. Model n_features is {0} and input n_features is {1} ",
                        this.nFeatures, n_features));
            }


            var proba = this.tree_.predict(X);

            var result = DenseMatrix.Create(n_samples, this.nOutputs, (x, y) => 0);
            for (int i = 0; i < proba.Length; i++)
            {
                result.SetColumn(i, proba[i].Column(0));
            }

            return result;
        }

        internal void CheckFitted()
        {
            if (this.tree_ == null)
            {
                throw new InvalidOperationException("Tree not initialized. Perform a fit first");
            }
        }

        /// <summary>
        /// Return the feature importances.
        ///
        /// The importance of a feature is computed as the (normalized) total
        /// reduction of the criterion brought by that feature.
        /// It is also known as the Gini importance.
        /// </summary>
        /// <returns>shape = [n_features]</returns>
        public Vector<double> FeatureImportances()
        {
            if (this.tree_ == null)
            {
                throw new InvalidOperationException("Estimator not fitted, call `fit` before `feature_importances_`.");
            }

            return this.tree_.ComputeFeatureImportances();
        }
    }
}
