// -----------------------------------------------------------------------
// <copyright file="LibSvmBase.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using System.Collections.Generic;
using System.Diagnostics;
using LibSvm;

namespace Sharpkit.Learn.Svm
{
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using System;
    using System.Linq;

    /// <summary>
    /// Base class for estimators that use libsvm as backing library
    /// 
    /// This implements support vector machine classification and regression.
    ///
    /// Parameter documentation is in the derived `SVC` class.
    /// </summary>
    public class LibSvmBase
    {
        private bool _sparse;
        public Kernel kernel { get; private set; }
        public int degree { get; private set; }
        public LibSvmImpl _impl { get; private set; }
        public double gamma { get; private set; }
        public double coef0 { get; private set; }
        public double tol { get; private set; }
        private double C;
        public double nu { get; private set; }
        public double epsilon { get; private set; }
        public bool shrinking { get; private set; }
        public bool probability { get; private set; }
        public int cache_size { get; private set; }
        private double[] class_weight;
        private bool verbose;
        private int max_iter;

        protected double _gamma;
        
        private int fit_status_;
        private Tuple<int, int> shape_fit_;
        public svm_model model { get; private set; }
        private Matrix<double> __Xfit;

        /// <summary>
        /// number of support vectors in each class.
        /// </summary>
        public int[] NSupport
        {
            get { return this.model.nSV; }
        }

        /// <summary>
        /// probability estimates, empty array for probability=False
        /// </summary>
        public double[] ProbA
        {
            get { return this.model.probA; }
        }

        /// <summary>
        /// probability estimates, empty array for probability=False
        /// </summary>
        public double[] ProbB
        {
            get { return this.model.probB; }
        }

        /// <summary>
        /// labels for different classes (only relevant in classification).
        /// </summary>
        public int[] Label
        {
            get { return this.model.label; }
        }

        /// <summary>
        /// coefficients of support vectors in decision function.
        /// </summary>
        public Matrix<double> DualCoef
        {
            get
            {
                return arrayToMatrix(this.model.sv_coef);
            }
        }

        private Matrix<double> arrayToMatrix(double[][] arr)
        {
            Matrix<double> m = this._sparse ? (Matrix<double>)new SparseMatrix(arr.Length, arr[0].Length) : new DenseMatrix(arr.Length, arr[0].Length);
            for (int i = 0; i < arr.Length; i++)
                for (int j = 0; j < arr[i].Length; j++)
                    m[i, j] = arr[i][j];
            return m;
        }

        /// <summary>
        /// [n_support] index of support vectors
        /// </summary>
        public int[] Support
        {
            get { return model.sv_indices; }
        }

        /// <summary>
        /// [n_support, n_features]
        /// support vectors (equivalent to X[support]). Will return an
        /// empty array in the case of precomputed kernel.
        /// </summary>
        public Matrix<double> SupportVectors
        {
            get
            {
                var nVectors = model.SV.Length;
                var nFeatures = this.shape_fit_.Item2;
                Matrix<double> m = this._sparse ? (Matrix<double>)new SparseMatrix(nVectors, nFeatures) : new DenseMatrix(nVectors, nFeatures);
                
                for (int i = 0; i < nVectors; i++)
                {
                    for (int j = this.kernel.KernelType == SparseKernel.Precomputed ? 1 : 0; j < model.SV[i].Length; j++)
                    {
                        m[i, model.SV[i][j].index] = model.SV[i][j].value;
                    }
                }

                return m;
            }
        }

        /// <summary>
        /// intercept in decision function
        /// </summary>
        public Vector<double> Intercept
        {
            get { return this.model.rho.ToDenseVector() * -1; }
        }

        public LibSvmBase(LibSvmImpl impl, Kernel kernel, int degree, double gamma, double coef0,
                 double tol, double C, double nu, double epsilon, bool shrinking, bool probability, int cache_size,
                 double[] classWeight, bool verbose, int max_iter)
        {
            this._impl = impl;
            this.kernel = kernel;
            this.degree = degree;
            this.gamma = gamma;
            this.coef0 = coef0;
            this.tol = tol;
            this.C = C;
            this.nu = nu;
            this.epsilon = epsilon;
            this.shrinking = shrinking;
            this.probability = probability;
            this.cache_size = cache_size;
            this.class_weight = class_weight;
            this.verbose = verbose;
            this.max_iter = max_iter;
        }
    
        public bool Pairwise
        {
            get { return this.kernel.KernelType == SparseKernel.Precomputed || kernel.KernelFunction != null; }
        }
        

        /// <summary>
        /// Fit the SVM model according to the given training data.
        /// </summary>
        /// <param name="X">[n_samples, n_features]
        ///    Training vectors, where n_samples is the number of samples
        ///    and n_features is the number of features.</param>
        /// <param name="y">[n_samples]
        ///    Target values (class labels in classification, real numbers in
        ///    regression)</param>
        /// <param name="sample_weight">[n_samples], optional
        ///    Weights applied to individual samples (1. for unweighted).</param>
        /// <remarks>
        /// If X and y are not C-ordered and contiguous arrays of np.float64 and
        /// X is not a scipy.sparse.csr_matrix, X and/or y may be copied.
        ///
        /// If X is a dense array, then the other methods will not support sparse
        /// matrices as input.
        /// </remarks>
        public virtual void Fit(Matrix<double> X, Vector<double> y, double[] sample_weight= null)
        {
            this._sparse = X is SparseMatrix && !this.Pairwise;

            if (this._sparse && this.Pairwise)
            {
                throw new ArgumentException
                    ("Sparse precomputed kernels are not supported. " +
                     "Using sparse data and dense kernels is possible " +
                     "by not using the ``sparse`` parameter");
            }

            sample_weight = sample_weight ?? new double[0];

            // input validation
            if (this._impl != LibSvmImpl.one_class && X.RowCount != y.Count)
            {
                throw new ArgumentException
                    (string.Format("X and y have incompatible shapes.\n" +
                                   "X has {0} samples, but y has {1}.", X.RowCount, y.Count));
            }

            if (this.kernel.KernelType == SparseKernel.Precomputed && this.kernel.KernelFunction == null && X.RowCount != X.ColumnCount)
            {
                throw new ArgumentException("X.RowCount should be equal to X.ColumnCount");
            }

            if (sample_weight.Length > 0 && sample_weight.Length != X.ColumnCount)
            {
                throw new ArgumentException(string.Format("sample_weight and X have incompatible shapes:" +
                                                          "{0} vs {1}\n" +
                                                          "Note: Sparse matrices cannot be indexed w/" +
                                                          "boolean masks (use `indices=True` in CV).",
                                                          sample_weight.Length, X.ColumnCount));
            }

            if (new[] {SparseKernel.Poly, SparseKernel.Rbf}.Contains(this.kernel.KernelType) && this.gamma == 0)
            {
                // if custom gamma is not provided ...
                this._gamma = 1.0/X.ColumnCount;
            }
            else
            {
                this._gamma = this.gamma;
            }

            if (this.verbose) // pragma: no cover
            {
                Console.WriteLine("[LibSVM]");
            }

            this.fit(X, y, sample_weight, this._impl);
            this.shape_fit_ = X.Shape();
        }

    /// <summary>
    /// Validation of y and class_weight.
    ///
    ///   Default implementation for SVR and one-class; overridden in BaseSVC.
    /// </summary>
    /// <param name="y"></param>
    protected virtual Vector<double> _validate_targets(Vector<double> y)
    {
        // XXX this is ugly.
        // Regression models should not have a class_weight_ attribute.
        return y;
    }

    private List<string> Warnings = new List<string>();
    private void _warn_from_fit_status()
    {
        Debug.Assert(this.fit_status_ == 0 || this.fit_status_ == 1);
        if (this.fit_status_ == 1)
        {
            Warnings.Add(string.Format("Solver terminated early (max_iter=%i)." +
            "  Consider pre-processing your data with" +
            " StandardScaler or MinMaxScaler.",
            this.max_iter));
        }
    }
    
    private void fit(Matrix<double> X, Vector<double> y, double[] sample_weight, LibSvmImpl solver_type)
    {
        if (this.kernel.KernelFunction != null)
        {
            // you must store a reference to X to compute the kernel in predict
            // TODO: add keyword copy to copy on demand
            this.__Xfit = X;
            X = this._compute_kernel(X);

            if (X.RowCount != X.ColumnCount)
            {
                throw new ArgumentException("X.RowCount should be equal to X.ColumnCount");
            }
        }

        //libsvm.svm.set_verbosity_wrap(self.verbose)

        // we don't pass **self.get_params() to allow subclasses to
        // add other parameters to __init__
        //self.support_, self.support_vectors_, self.n_support_, \
        //    self.dual_coef_, self.intercept_, self._label, self.probA_, \
        //    self.probB_, self.fit_status_ = 
        
        svm_problem problem = new svm_problem();
        problem.l = X.RowCount;
        problem.x = new svm_node[X.RowCount][];
        foreach (var row in X.RowEnumerator())
        {
            if (kernel.KernelType == SparseKernel.Precomputed)
            {
                var svmNodes =
                    row.Item2.GetIndexedEnumerator().Select(i => new svm_node {index = i.Item1 + 1, value = i.Item2});
                problem.x[row.Item1] =
                    new[] {new svm_node {index = 0, value = row.Item1 + 1}}.Concat(svmNodes).ToArray();
            }
            else
            {
                var svmNodes =
                    row.Item2.GetIndexedEnumerator().Select(i => new svm_node { index = i.Item1, value = i.Item2 });
                problem.x[row.Item1] = svmNodes.ToArray();
            }
        }
        problem.y = y.ToArray();
        //todo:
        //problem.w

         svm_parameter prm = new svm_parameter();
         prm.svm_type = (int)this._impl;
         prm.kernel_type = (int)this.kernel.KernelType;
         prm.degree = degree;
         prm.coef0 = coef0;
         prm.nu = nu;
         prm.cache_size = cache_size;
         prm.C = C;
         prm.eps = this.tol;
         prm.p = this.epsilon;
         prm.shrinking = shrinking ? 1 : 0;
         prm.probability = probability ? 1: 0;
         //todo:
         prm.nr_weight = 0;// this.class_weight == null ? 0 : class_weight.Length;
         //prm.weight_label = Enumerable.Range(0, this.class_weight.Length).ToArray();
         //prm.weight = this.class_weight;

         prm.gamma = _gamma;

        this.model = svm.svm_train(problem, prm);

        this._warn_from_fit_status();
    }

    /*
        private void _sparse_fit(Matrix<double> X, Vector<double> y,double[] sample_weight, LibSvmImpl solver_type, SparseKernel kernel)
    {
        //X.data = np.asarray(X.data, dtype=np.float64, order='C')
        //X.sort_indices()

        //kernel_type = self._sparse_kernels.index(kernel)

        //libsvm_sparse.set_verbosity_wrap(self.verbose)

        self.support_, self.support_vectors_, dual_coef_data, \
            self.intercept_, self._label, self.n_support_, \
            self.probA_, self.probB_, self.fit_status_ = \
            libsvm_sparse.libsvm_sparse_train(
                X.shape[1], X.data, X.indices, X.indptr, y, solver_type,
                kernel_type, self.degree, self._gamma, self.coef0, self.tol,
                self.C, self.class_weight_,
                sample_weight, self.nu, self.cache_size, self.epsilon,
                int(self.shrinking), int(self.probability), self.max_iter)

        self._warn_from_fit_status()

        n_class = len(self._label) - 1
        n_SV = self.support_vectors_.shape[0]

        dual_coef_indices = np.tile(np.arange(n_SV), n_class)
        dual_coef_indptr = np.arange(0, dual_coef_indices.size + 1,
                                     dual_coef_indices.size / n_class)
        self.dual_coef_ = sp.csr_matrix(
            (dual_coef_data, dual_coef_indices, dual_coef_indptr),
            (n_class, n_SV))
    }
     * */

    /// <summary>
    /// Perform regression on samples in X.
    ///
    ///    For an one-class model, +1 or -1 is returned.
    /// </summary>
    /// <param name="X">[n_samples, n_features]</param>
    /// <returns>[n_samples]</returns>
    public Vector<double> predict(Matrix<double> X)
    {
        X = this._validate_for_predict(X);
        return this.predict_(X);
    }

    private Vector<double> predict_(Matrix<double> X)
    {
        int n_samples = X.RowCount;
        int n_features = X.ColumnCount;
        X = this._compute_kernel(X);

        if (this.kernel.KernelFunction != null)
        {
            if (X.ColumnCount != this.shape_fit_.Item1)
            {
                throw new ArgumentException(string.Format("X.shape[1] ={0} should be equal to {1}, " +
                                            "the number of samples at training time",
                                            X.ColumnCount, this.shape_fit_.Item1));
            }
        }
        var C = 0.0;  // C is not useful here

         svm_parameter prm = new svm_parameter();
         prm.svm_type = (int)this._impl;
         prm.kernel_type = (int)this.kernel.KernelType;
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

        DenseVector result = new DenseVector(X.RowCount);
        foreach (var r in X.RowEnumerator())
        {
            svm_node[] svmNodes;
            if (this.kernel.KernelType == SparseKernel.Precomputed)
            {
                svmNodes = r.Item2.Select(
                    (v, i) => new svm_node { index = i + 1, value = v }).ToArray();
                svmNodes = new[] {new svm_node {index = 0, value = 1}}.Concat(svmNodes).ToArray();
            }
            else
            {
                svmNodes = r.Item2.Select((v, i) => new svm_node { index = i, value = v }).ToArray();
            }

            result[r.Item1] = svm.svm_predict(m, svmNodes);
        }

        return result;
    }

    /*
        private Vector<double> _sparse_predict(Matrix<double> X)
    {
        X = sp.csr_matrix(X, dtype=np.float64)

        kernel = self.kernel;
        if (callable(kernel))
        {
            kernel = "precomputed";
        }

        kernel_type = self._sparse_kernels.index(kernel);

        var C = 0.0;  // C is not useful here

        return libsvm_sparse.libsvm_sparse_predict(
            X.data, X.indices, X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self.dual_coef_.data, self._intercept_,
            LIBSVM_IMPL.index(self._impl), kernel_type,
            self.degree, self._gamma, self.coef0, self.tol,
            C, self.class_weight_,
            self.nu, self.epsilon, self.shrinking,
            self.probability, self.n_support_, self._label,
            self.probA_, self.probB_);
        throw new NotImplementedException();
    }
     */

    /// <summary>
    /// Return the data transformed by a callable kernel.
    /// </summary>
    /// <param name="X"></param>
    protected Matrix<double> _compute_kernel(Matrix<double> X)
    {
        if (this.kernel.KernelFunction != null)
        {
            // in the case of precomputed kernel given as a function, we
            // have to compute explicitly the kernel matrix
            var vk = kernel.KernelFunction(X, this.__Xfit);
            if (vk is SparseMatrix)
                vk = DenseMatrix.OfMatrix(vk);

            return vk;
        }

        return X;
    }

    /// <summary>
    /// Distance of the samples X to the separating hyperplane.
    /// </summary>
    /// <param name="X">[n_samples, n_features]</param>
    /// <returns>
    /// [n_samples, n_class * (n_class-1) / 2]
    ///        Returns the decision function of the sample for each class
    ///        in the model.
    /// </returns>
        public Matrix<double> decision_function(Matrix<double> X)
    {
        /*
            if (this._sparse)
        {
            throw new NotImplementedException("Decision_function not supported for sparse SVM.");
        }

        X = this._validate_for_predict(X);
        X = this._compute_kernel(X);

        var C = 0.0;  // C is not useful here

        kernel = self.kernel;
        if (callable(kernel))
        {
            kernel = 'precomputed'
        }

        dec_func = libsvm.decision_function(
            X, self.support_, self.support_vectors_, self.n_support_,
            self.dual_coef_, self._intercept_, self._label,
            self.probA_, self.probB_,
            svm_type=LIBSVM_IMPL.index(self._impl),
            kernel=kernel, C=C, nu=self.nu,
            probability=self.probability, degree=self.degree,
            shrinking=self.shrinking, tol=self.tol, cache_size=self.cache_size,
            coef0=self.coef0, gamma=self._gamma, epsilon=self.epsilon)

        // In binary case, we need to flip the sign of coef, intercept and
        // decision function.
        if (self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2)
        {
            return -dec_func
        }

        return dec_func
         * */
        throw new NotImplementedException();
    }

    protected Matrix<double> _validate_for_predict(Matrix<double> X)
    {
        if (this._sparse && !(X is SparseMatrix))
        {
            X = SparseMatrix.OfMatrix(X);
        }

        /*if (self._sparse)
        {
            X.sort_indices();
        }*/

        if ((X is SparseMatrix) && !this._sparse && this.kernel.KernelFunction == null)
        {
            throw new ArgumentException(string.Format(
                "cannot use sparse input in {0} trained on dense data", this.GetType().Name));

        }
        
        int n_samples = X.RowCount;
        int n_features = X.ColumnCount;

        if (this.kernel.KernelType == SparseKernel.Precomputed && this.kernel.KernelFunction == null)
        {
            if (X.ColumnCount != this.shape_fit_.Item1)
            {
                throw new ArgumentException(string.Format("X.shape[1] = {0} should be equal to {1}, " +
                                                          "the number of samples at training time",
                                                          X.ColumnCount,
                                                          this.shape_fit_.Item1));
            }
        }
        else if (n_features != this.shape_fit_.Item2)
        {
            throw new ArgumentException(string.Format("X.shape[1] = {0} should be equal to {1}, " +
                                        "the number of features at training time",
                                        n_features, this.shape_fit_.Item2));
        }

        return X;
    }


    public Matrix<double> Coef
    {
        get
        {
            if (this.kernel.KernelType != SparseKernel.Linear)
                throw new ArgumentException("coef is only available when using a linear kernel");

            if (this.DualCoef.RowCount == 1)
            {
                // binary classifier
                return this.DualCoef * this.SupportVectors * -1;
            }
            else
            {
                // 1vs1 classifier
                var coef = _one_vs_one_coef(this.DualCoef, this.NSupport,
                                        this.SupportVectors);
                if (coef[0] is SparseVector)
                {
                    return SparseMatrix.OfRows(coef.Length, coef[0].Count, coef);
                }
                else
                {
                    return DenseMatrix.OfRows(coef.Length, coef[0].Count, coef);
                }
            }
        }
    }

        private int[] cumsum(int[] a)
        {
            int[] result = new int[a.Length];
            int s = 0;
            for (int i=0; i<a.Length; i++)
            {
                s += a[i];
                result[i] = s;
            }

            return result;
        }

        /// <summary>
        /// Generate primal coefficients from dual coefficients
        /// for the one-vs-one multi class LibSVM in the case
        /// of a linear kernel.
        /// </summary>
        /// <param name="?"></param>
        /// <param name="?"></param>
        /// <param name="?"></param>
        public Vector<double>[] _one_vs_one_coef(Matrix<double> dual_coef, int[] n_support, Matrix<double> support_vectors)
        {
            // get 1vs1 weights for all n*(n-1) classifiers.
            // this is somewhat messy.
            // shape of dual_coef_ is nSV * (n_classes -1)
            // see docs for details
            int n_class = dual_coef.RowCount + 1;

            // XXX we could do preallocation of coef but
            // would have to take care in the sparse case
            var coef = new List<Vector<double>>();
            var sv_locs = cumsum(new[] {0}.Concat(n_support).ToArray());
            for (int class1 =0; class1 < n_class; class1++)
            {
                // SVs for class1:
                var sv1 = support_vectors.SubMatrix(sv_locs[class1], sv_locs[class1 + 1] - sv_locs[class1], 0,
                                                support_vectors.ColumnCount);
                for (int class2 = class1+1; class2 < n_class; class2++)
                {
                    // SVs for class1:
                    var sv2 = support_vectors.SubMatrix(sv_locs[class2], sv_locs[class2 + 1] - sv_locs[class2], 0,
                                                        support_vectors.ColumnCount);

                    // dual coef for class1 SVs:
                    var alpha1 = dual_coef.Row(class2 - 1).SubVector(sv_locs[class1], sv_locs[class1 + 1] - sv_locs[class1]);
                    // dual coef for class2 SVs:
                    var alpha2 = dual_coef.Row(class1).SubVector(sv_locs[class2], sv_locs[class2 + 1] - sv_locs[class2]);
                    // build weight for class1 vs class2

                    coef.Add(alpha1*sv1 + alpha2*sv2);
                }
            }

            return coef.ToArray();
        }
    }
}
