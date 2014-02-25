// -----------------------------------------------------------------------
// <copyright file="SvcBase.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Svm
{
    using System;
    using System.Linq;
    using LibSvm;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using Sharpkit.Learn.Preprocessing;

    /// <summary>
    /// ABC for LibSVM-based classifiers.
    /// </summary>
    /// <typeparam name="TLabel">Type of class label.</typeparam>
    public class SvcBase<TLabel> : LibSvmBase, IClassifier<TLabel>
    {
        /// <summary>
        /// Label encoder.
        /// </summary>
        private LabelEncoder<TLabel> enc;
        
        internal SvcBase(
            LibSvmImpl impl,
            Kernel kernel,
            int degree,
            double gamma,
            double coef0,
            double tol,
            double c,
            double nu,
            double epsilon,
            bool shrinking,
            bool probability,
            int cacheSize,
            ClassWeightEstimator<TLabel> classWeightEstimator,
            bool verbose) :
            base(
                impl,
                kernel,
                degree,
                gamma,
                coef0,
                tol,
                c,
                nu,
                epsilon,
                shrinking,
                probability,
                cacheSize,
                verbose)
        {
            this.ClassWeightEstimator = classWeightEstimator;
        }

        /// <summary>
        /// Gets or sets class weight estimator.
        /// </summary>
        public ClassWeightEstimator<TLabel> ClassWeightEstimator { get; set; }

        /// <summary>
        /// Gets array with labels for each class.
        /// </summary>
        public TLabel[] Classes
        {
            get { return this.enc.Classes; }
        }

        /// <summary>
        /// Gets intercept in decision function
        /// </summary>
        public override Vector<double> Intercept
        {
            get
            {
                if (new[] { LibSvmImpl.c_svc, LibSvmImpl.nu_svc }.Contains(this.Impl) && (this.Classes.Length == 2))
                {
                    return base.Intercept * -1;
                }

                return base.Intercept;
            }
        }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Training vectors,
        /// where nSamples is the number of samples and nFeatures
        /// is the number of features.</param>
        /// <param name="y">[nSamples] Target class labels.</param>
        /// <param name="sampleWeight">Sample weights.</param>
        /// <returns>Reference to itself.</returns>
        public virtual void Fit(Matrix<double> x, TLabel[] y, Vector<double> sampleWeight = null)
        {
            if (sampleWeight != null)
            {
                throw new ArgumentException("Sample weights are not supported by the classifier");
            }

            this.enc = new LabelEncoder<TLabel>();
            int[] y1 = this.enc.FitTransform(y);
            if (this.Classes.Length < 2)
            {
                throw new ArgumentException("The number of classes has to be greater than one.");
            }

            this.ClassWeight = (this.ClassWeightEstimator ?? ClassWeightEstimator<TLabel>.Auto)
                .ComputeWeights(this.enc.Classes, y1)
                .ToArray();

            base.Fit(x, DenseVector.OfEnumerable(y1.Select(Convert.ToDouble)));
        }

        /// <summary>
        /// Perform classification on samples in X.
        /// For an one-class model, +1 or -1 is returned.
        /// </summary>
        /// <param name="x">[n_samples, n_features]</param>
        /// <returns>Class labels for samples in X.</returns>
        public TLabel[] Predict(Matrix<double> x)
        {
            var y = this.predict(x);
            return y.Select(v => this.Classes[(int)v]).ToArray();
        }

        /// <summary>
        /// <para>
        /// Compute probabilities of possible outcomes for samples in X.
        /// </para>
        /// <para>
        /// The model need to have probability information computed at training
        /// time: fit with attribute <see cref="LibSvmBase.Probability"/> set to True.
        /// </para>
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]</param>
        /// <returns>[nSamples, nClasses]
        ///    Returns the probability of the sample for each class in
        ///   the model. The columns correspond to the classes in sorted
        ///    order, as they appear in the attribute `classes_`.</returns>
        /// <remarks>
        ///         The probability model is created using cross validation, so
        /// the results can be slightly different than those obtained by
        /// predict. Also, it will produce meaningless results on very small
        /// datasets.
        /// </remarks>
        public Matrix<double> PredictProba(Matrix<double> x)
        {
            if (!this.Probability)
            {
                throw new InvalidOperationException("probability estimates must be enabled to use this method");
            }

            if (!new[] { LibSvmImpl.c_svc, LibSvmImpl.nu_svc }.Contains(this.Impl))
            {
                throw new NotImplementedException("predict_proba only implemented for SVC and NuSVC");
            }

            x = this.ValidateForPredict(x);
            return this.PredictProbaInternal(x);
        }

        /// <summary>
        /// Distance of the samples X to the separating hyperplane.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]</param>
        /// <returns>
        /// [nSamples, nClass * (nClass-1) / 2]
        ///        Returns the decision function of the sample for each class
        ///        in the model.
        /// </returns>
        public Matrix<double> DecisionFunction(Matrix<double> x)
        {
            if (this.IsSparse)
            {
                throw new NotImplementedException("Decision_function not supported for sparse SVM.");
            }

            x = this.ValidateForPredict(x);
            x = this.ComputeKernel(x);

            DenseMatrix result;
            if (new[] { LibSvmImpl.c_svc, LibSvmImpl.nu_svc }.Contains(this.Impl) && this.Classes.Length == 2)
            {
                result = new DenseMatrix(x.RowCount, this.Classes.Length > 2 ? this.Classes.Length : 1);
            }
            else
            {
                result = new DenseMatrix(x.RowCount, this.Classes.Length);
            }

            foreach (var r in x.RowEnumerator())
            {
                double[] darr = new double[result.ColumnCount];
                var svmNodes = r.Item2.Select((v, i) => new svm_node { index = i, value = v }).ToArray();
                svm.svm_predict_values(this.Model, svmNodes, darr);
                result.SetRow(r.Item1, darr);
            }

            // In binary case, we need to flip the sign of coef, intercept and
            // decision function.
            if (new[] { LibSvmImpl.c_svc, LibSvmImpl.nu_svc }.Contains(this.Impl) && this.Classes.Length == 2)
            {
                return result * -1;
            }

            return result;
        }

        private Matrix<double> PredictProbaInternal(Matrix<double> x)
        {
            x = this.ComputeKernel(x);

            DenseMatrix result = new DenseMatrix(x.RowCount, this.Classes.Length);
            foreach (var r in x.RowEnumerator())
            {
                svm_node[] svmNodes;
                if (Kernel.LibSvmKernel == LibSvmKernel.Precomputed)
                {
                    svmNodes = r.Item2.Select(
                        (v, i) => new svm_node { index = i + 1, value = v }).ToArray();
                    svmNodes = new[] { new svm_node { index = 0, value = 1 } }.Concat(svmNodes).ToArray();
                }
                else
                {
                    svmNodes =
                        r.Item2.GetIndexedEnumerator()
                        .Select(i => new svm_node { index = i.Item1, value = i.Item2 }).ToArray();
                }

                double[] darr = new double[this.Classes.Length];
                var res = svm.svm_predict_probability(this.Model, svmNodes, darr);
                result.SetRow(r.Item1, darr);
            }

            return result;
        }
    }
}
