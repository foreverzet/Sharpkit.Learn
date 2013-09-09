// -----------------------------------------------------------------------
// <copyright file="SvcBase.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using LibSvm;

namespace Sharpkit.Learn.Svm
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using Sharpkit.Learn.Preprocessing;


    /// <summary>
    /// ABC for LibSVM-based classifiers.
    /// </summary>
    public class SvcBase<TLabel> : LibSvmBase
    {
        private LabelEncoder<TLabel> _enc;
        

        public SvcBase(LibSvmImpl impl, Kernel kernel, int degree, double gamma, double coef0, double tol, double C, double nu, double epsilon, bool shrinking, bool probability, int cache_size, ClassWeight<TLabel> classWeight, bool verbose, int max_iter) : base(impl, kernel, degree, gamma, coef0, tol, C, nu, epsilon, shrinking, probability, cache_size, /*todo*/null, verbose, max_iter)
        {
        }

        public TLabel[] Classes
        {
            get { return this._enc.Classes; }
        }

        public void Fit(Matrix<double> X, TLabel[] y, Vector<double> sample_weight = null)
        {
            this._enc = new LabelEncoder<TLabel>();
            int[] y1 = this._enc.FitTransform(y);
            if (this.Classes.Length < 2)
                throw new ArgumentException("The number of classes has to be greater than one.");

            base.Fit(X, DenseVector.OfEnumerable(y1.Select(Convert.ToDouble)), sample_weight == null ? null : sample_weight.ToArray());

                        // todo:
            // In binary case, we need to flip the sign of coef, intercept and
            // decision function. Use self._intercept_ internally.
            //this._intercept_ = this.intercept_.Clone();
            //if (new[]{LibSvmImpl.c_svc, LibSvmImpl.nu_svc}.Contains(this.Impl) && (this.Classes.Length == 2))
            //{
            //    this.Intercept *= -1;
            //}

        }

        /// <summary>
        /// Perform classification on samples in X.
        /// For an one-class model, +1 or -1 is returned.
        /// </summary>
        /// <param name="X">[n_samples, n_features]</param>
        /// <returns>Class labels for samples in X.</returns>
        public TLabel[] Predict(Matrix<double> X)
        {
            var y = base.predict(X);
            return y.Select(v => this.Classes[(int)v]).ToArray();
        }

        /// <summary>
        /// Compute probabilities of possible outcomes for samples in X.
        /// 
        /// The model need to have probability information computed at training
        /// time: fit with attribute `probability` set to True.
        /// </summary>
        /// <param name="X">[n_samples, n_features]</param>
        /// <returns>[n_samples, n_classes]
        ///    Returns the probability of the sample for each class in
        ///   the model. The columns correspond to the classes in sorted
        ///    order, as they appear in the attribute `classes_`.</returns>
        /// <remarks>
        ///         The probability model is created using cross validation, so
        /// the results can be slightly different than those obtained by
        /// predict. Also, it will produce meaningless results on very small
        /// datasets.
        /// </remarks>
        public Matrix<double> predict_proba(Matrix<double> X)
        {
            if (!this.probability)
            {
                throw new InvalidOperationException("probability estimates must be enabled to use this method");
            }

            if (!new[] { LibSvmImpl.c_svc, LibSvmImpl.nu_svc }.Contains(this._impl))
            {
                throw new NotImplementedException("predict_proba only implemented for SVC and NuSVC");
            }

            X = this._validate_for_predict(X);
            return this._predict_proba(X);
        }

        public Matrix<double> _predict_proba(Matrix<double> X)
        {
            X = this._compute_kernel(X);

            var C = 0.0;

            var kernelType = this.kernel.KernelType;
            if (kernel.KernelFunction != null)
                kernelType = SparseKernel.Precomputed;

            double[] prob_estimates = new double[this.Classes.Length];

             svm_parameter prm = new svm_parameter();
             prm.svm_type = (int)this._impl;
             prm.kernel_type = (int)kernelType;
             prm.degree = degree;
             prm.coef0 = coef0;
             prm.nu = nu;
             prm.cache_size = cache_size;
             prm.C = C;
             prm.eps = this.tol;
             prm.p = this.epsilon;
             prm.shrinking = shrinking ? 1 :0;
             prm.probability = probability ? 1: 0;
            //todo:
             prm.nr_weight = 0;// this.class_weight.Length;
            prm.weight_label = null;// Enumerable.Range(0, this.class_weight.Length).ToArray();
            prm.weight = null;// this.class_weight;
             prm.gamma = _gamma;
             //prm.max_iter = max_iter;

            svm_model m = new svm_model();
            m.param = prm;
            m.l = model.l;
            m.label = model.label;
            m.nr_class = model.nr_class;
            m.nSV = model.nSV;
            m.probA = model.probA;
            m.probB = model.probB;
            m.rho = model.rho;
            m.SV = model.SV;
            m.sv_coef = model.sv_coef;
            m.sv_indices = model.sv_indices;

            DenseMatrix result = new DenseMatrix(X.RowCount, this.Classes.Length);
            foreach (var r in X.RowEnumerator())
            {
                svm_node n = new svm_node();
                double[] darr = new double[this.Classes.Length];
                var res = svm.svm_predict_probability(model,
                                                         r.Item2.GetIndexedEnumerator().Select(
                                                             v => new svm_node {index = v.Item1, value = v.Item2}).ToArray(), darr);
                result.SetRow(r.Item1, darr);
            }

            return result;
        }

        /*
        private Matrix<double> _sparse_predict_proba(Matrix<double> X)
        {
            
            X.data = np.asarray(X.data, dtype = np.float64, order = 'C')

            kernel = self.kernel
            if callable(kernel)
            {
                kernel = 'precomputed'
            }

            kernel_type = self._sparse_kernels.index(kernel)

            return libsvm.svm.svm_predict_probability(
                X.data, X.indices, X.indptr,
                self.support_vectors_.data,
                self.support_vectors_.indices,
                self.support_vectors_.indptr,
                self.dual_coef_.data, self._intercept_,
                LIBSVM_IMPL.index(self._impl), kernel_type,
                self.degree, self._gamma, self.coef0, self.tol,
                self.C, self.class_weight_,
                self.nu, self.epsilon, self.shrinking,
                self.probability, self.n_support_, self._label,
                self.probA_, self.probB_))
             * 
            throw new NotImplementedException();
        }*/

        
        /// <summary>
        /// Compute log probabilities of possible outcomes for samples in X.
        /// The model need to have probability information computed at training
        /// time: fit with attribute `probability` set to True.
        /// </summary>
        /// <param name="X">[n_samples, n_features]</param>
        /// <returns>[n_samples, n_classes]
        ///    Returns the log-probabilities of the sample for each class in
        ///    the model. The columns correspond to the classes in sorted
        ///    order, as they appear in the attribute `classes_`.</returns>
        /// <remarks>
        ///         The probability model is created using cross validation, so
        /// the results can be slightly different than those obtained by
        /// predict. Also, it will produce meaningless results on very small
        /// datasets.
        /// </remarks>
        public Matrix<double> predict_log_proba(Matrix<double> X)
        {
            throw new NotImplementedException();
            //return np.log(self.predict_proba(X));
        }
    }
}
