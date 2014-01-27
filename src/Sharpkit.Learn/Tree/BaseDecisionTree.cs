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
        private readonly int? max_depth;
        private int min_samples_split;
        private readonly int min_samples_leaf;
        private readonly MaxFeaturesChoice max_features;
        private readonly Random random_state;
        internal int n_features_;
        internal Tree tree_;
        private int n_outputs_;
        internal int max_features_;
        internal List<uint> n_classes_;
        internal List<TLabel> classes_;

        public BaseDecisionTree(
            Criterion criterion,
            Splitter splitter,
            int? max_depth,
            int min_samples_split,
            int min_samples_leaf,
            MaxFeaturesChoice max_features,
            Random random_state)
        {
            this.criterion = criterion;
            this.splitter = splitter;
            this.max_depth = max_depth;
            this.min_samples_split = min_samples_split;
            this.min_samples_leaf = min_samples_leaf;
            this.max_features = max_features;
            this.random_state = random_state ?? new Random();
        }

        /// <summary>
        /// Build a decision tree from the training set (X, y).
        /// </summary>
        /// <param name="X">[n_samples, n_features] The training input samples.</param>
        /// <param name="y">[n_samples, n_outputs] The target values.</param>
        /// <param name="sample_weight">[n_samples] or None.
        ///  Sample weights. If None, then samples are equally weighted. Splits
        ///  that would create child nodes with net zero or negative weight are
        ///  ignored while searching for a split in each node. In the case of
        ///  classification, splits are also ignored if they would result in any
        ///  single class carrying a negative weight in either child node.</param>
        /// <returns></returns>
        internal BaseDecisionTree<TLabel> fitRegression(Matrix<double> X, Matrix<double> y, Vector<double> sample_weight = null)
        {
            // Determine output settings
            int n_samples = X.RowCount;
            this.n_features_ = X.ColumnCount;


            if (y.RowCount != n_samples)
            {
                throw new ArgumentException(string.Format("Number of labels={0} does not match number of samples={1}",
                                                          y.RowCount,
                                                          n_samples));
            }

            //if (y.ndim == 1)
            //    # reshape is necessary to preserve the data contiguity against vs
            //    # [:, np.newaxis] that does not.
            //    y = np.reshape(y, (-1, 1))


            this.n_outputs_ = y.ColumnCount;
            //this.n_outputs_ = 1;
            int[] y_;

            this.classes_ = Enumerable.Repeat(default(TLabel), this.n_outputs_).ToList();
            this.n_classes_ = Enumerable.Repeat(1U, this.n_outputs_).ToList();


            //# Check parameters
            fitCommon1(X, y, n_samples, sample_weight, false);


            return this;
        }

        private void fitCommon1(Matrix<double> X, Matrix<double> y, int n_samples, Vector<double> sample_weight, bool isClassification)
        {
            int max_depth = this.max_depth ?? int.MaxValue;

            if (this.max_features == null)
            {
                this.max_features_ = this.n_features_;
            }
            else
            {
                this.max_features_ = max_features.ComputeMaxFeatures(this.n_features_, isClassification);
            }

            if (this.min_samples_split <= 0)
            {
                throw new ArgumentException("min_samples_split must be greater than zero.");
            }
            if (this.min_samples_leaf <= 0)
            {
                throw new ArgumentException("min_samples_leaf must be greater than zero.");
            }

            if (max_depth <= 0)
            {
                throw new ArgumentException("max_depth must be greater than zero. ");
            }

            if (!(0 < max_features_ && max_features_ <= this.n_features_))
            {
                throw new ArgumentException("max_features must be in (0, n_features]");
            }

            if (sample_weight != null)
            {
                if (sample_weight.Count != n_samples)
                {
                    throw new ArgumentException(
                        string.Format("Number of weights={0} does not match number of samples={1}",
                                      sample_weight.Count, n_samples));
                }
            }


            // Set min_samples_split sensibly
            min_samples_split = Math.Max(this.min_samples_split,
                                         2*this.min_samples_leaf);


            // Build tree
            ICriterion criterion = null;
            switch (this.criterion)
            {
                case Criterion.Gini:
                    criterion = new Gini((uint)n_outputs_, n_classes_.ToArray());
                    break;
                case Criterion.Entropy:
                    criterion = new Entropy((uint)n_outputs_, n_classes_.ToArray());
                    break;
                case Criterion.Mse:
                    criterion = new MSE((uint)n_outputs_);
                    break;
                default:
                    throw new InvalidOperationException("Unknown criterion type");
            }

            SplitterBase splitter = null;
            switch (this.splitter)
            {
                case Splitter.Best:
                    splitter = new BestSplitter(criterion, (uint)this.max_features_, (uint)this.min_samples_leaf, random_state);
                    break;
                case Splitter.PresortBest:
                    splitter = new PresortBestSplitter(criterion, (uint)this.max_features_, (uint)this.min_samples_leaf, random_state);
                    break;
                case Splitter.Random:
                    splitter = new RandomSplitter(criterion, (uint)this.max_features_, (uint)this.min_samples_leaf, random_state);
                    break;
                default:
                    throw new InvalidOperationException("Unknown splitter type");
            }

            this.tree_ = new Tree(this.n_features_, this.n_classes_.ToArray(),
                                  this.n_outputs_, splitter, (uint)max_depth,
                                  (uint)min_samples_split, (uint)this.min_samples_leaf);


            this.tree_.build(X, y, sampleWeight: sample_weight);
        }

        /// <summary>
        /// Build a decision tree from the training set (X, y).
        /// </summary>
        /// <param name="X">[n_samples, n_features] The training input samples. </param>
        /// <param name="y"> [n_samples] The target values.</param>
        /// <param name="sample_weight">[n_samples] or None
        /// Sample weights. If None, then samples are equally weighted. Splits
        /// that would create child nodes with net zero or negative weight are
        /// ignored while searching for a split in each node. In the case of
        /// classification, splits are also ignored if they would result in any
        /// single class carrying a negative weight in either child node.</param>
        /// <returns>
        /// Returns this.
        /// </returns>
        internal BaseDecisionTree<TLabel> fitClassification(
            Matrix<double> X,
            TLabel[] y,
            Vector<double> sample_weight = null)
        {
            // Determine output settings
            int n_samples = X.RowCount;
            this.n_features_ = X.ColumnCount;

            if (y.Length != n_samples)
            {
                throw new ArgumentException(string.Format("Number of labels={0} does not match number of samples={1}",
                                                          y.Length,
                                                          n_samples));
            }

            this.n_outputs_ = 1;
            int[] y_;


            this.classes_ = new List<TLabel>();
            this.n_classes_ = new List<uint>();

            var enc = new LabelEncoder<TLabel>();
            y_ = enc.FitTransform(y).ToArray();

            this.classes_.AddRange(enc.Classes);
            this.n_classes_.Add((uint)enc.Classes.Length);

            var yMatrix = y_.Select(v => (double)v).ToArray().ToDenseVector().ToColumnMatrix();
            fitCommon1(X, yMatrix, n_samples, sample_weight, true);


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


            if (this.n_features_ != n_features)
            {
                throw new ArgumentException(
                    String.Format(
                        "Number of features of the model must match the input. Model n_features is {0} and input n_features is {1} ",
                        this.n_features_, n_features));
            }


            Matrix<double> proba = this.tree_.predict(X)[0];

            return proba.ArgmaxColumns().Select(v => this.classes_[v]).ToArray();
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


            if (this.n_features_ != n_features)
            {
                throw new ArgumentException(
                    String.Format(
                        "Number of features of the model must match the input. Model n_features is {0} and input n_features is {1} ",
                        this.n_features_, n_features));
            }


            var proba = this.tree_.predict(X);

            var result = DenseMatrix.Create(n_samples, this.n_outputs_, (x, y) => 0);
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
