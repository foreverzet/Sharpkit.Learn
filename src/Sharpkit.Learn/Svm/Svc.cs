// -----------------------------------------------------------------------
// <copyright file="Svc.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Generic;

namespace Sharpkit.Learn.Svm
{
    using System;

    /// <summary>
    /// C-Support Vector Classification.
    /// 
    /// The implementations is a based on libsvm. The fit time complexity
    /// is more than quadratic with the number of samples which makes it hard
    /// to scale to dataset with more than a couple of 10000 samples.
    /// 
    /// The multiclass support is handled according to a one-vs-one scheme.
    /// 
    /// For details on the precise mathematical formulation of the provided
    /// kernel functions and how `gamma`, `coef0` and `degree` affect each,
    /// see the corresponding section in the narrative documentation:
    /// :ref:`svm_kernels`.
    ///
    /// .. The narrative documentation is available at http://scikit-learn.org/
    /// </summary>
    public class Svc<TLabel>
    {
        private SvcBase<TLabel> baseSvc;
        /*

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    degree : int, optional (default=3)
        Degree of kernel function.
        It is significant only in 'poly'.

    gamma : float, optional (default=0.0)
        Kernel coefficient for 'rbf', 'poly' and 'sigm'.
        If gamma is 0.0 then 1/n_features will be used instead.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    probability: boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling predict_proba.

    shrinking: boolean, optional (default=True)
        Whether to use the shrinking heuristic.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    cache_size : float, optional
        Specify the size of the kernel cache (in MB)

    class_weight : {dict, 'auto'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one. The 'auto' mode uses the values of y to
        automatically adjust weights inversely proportional to
        class frequencies.

    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    `support_` : array-like, shape = [n_SV]
        Index of support vectors.

    `support_vectors_` : array-like, shape = [n_SV, n_features]
        Support vectors.

    `n_support_` : array-like, dtype=int32, shape = [n_class]
        number of support vector for each class.

    `dual_coef_` : array, shape = [n_class-1, n_SV]
        Coefficients of the support vector in the decision function. \
        For multiclass, coefficient for all 1-vs-1 classifiers. \
        The layout of the coefficients in the multiclass case is somewhat \
        non-trivial. See the section about multi-class classification in the \
        SVM section of the User Guide for details.

    `coef_` : array, shape = [n_class-1, n_features]
        Weights asigned to the features (coefficients in the primal
        problem). This is only available in the case of linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`

    `intercept_` : array, shape = [n_class * (n_class-1) / 2]
        Constants in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.svm import SVC
    >>> clf = SVC()
    >>> clf.fit(X, y) #doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
            gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
            shrinking=True, tol=0.001, verbose=False)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]

    See also
    --------
    SVR
        Support Vector Machine for Regression implemented using libsvm.

    LinearSVC
        Scalable Linear Support Vector Machine for classififcation
        implemented using liblinear. Check the See also section of
        LinearSVC for more comparison element.

    """
    */

        public Svc(double C = 1.0, Kernel kernel = null, int degree = 3, double gamma = 0.0,
                   double coef0 = 0.0, bool shrinking = true, bool probability = false,
                   double tol = 1e-3, int cache_size = 200, ClassWeight<TLabel> class_weight = null,
                   bool verbose = false, int max_iter = -1)
        {
            kernel = kernel ?? Kernel.FromSparseKernel(SparseKernel.Rbf);
            this.baseSvc = new SvcBase<TLabel>(LibSvmImpl.c_svc, kernel, degree, gamma, coef0, tol, C, 0.0, 0.0,
                                               shrinking,
                                               probability, cache_size, class_weight, verbose, max_iter);
        }

        public Matrix<double> DualCoef
        {
            get { return this.baseSvc.DualCoef; }
        }

        /// <summary>
        /// Index of support vectors.
        /// </summary>
        public int[] Support
        {
            get { return this.baseSvc.Support.Select( s=> s -1).ToArray(); }
        }

        /// <summary>
        /// [n_SV, n_features] Support vectors.
        /// </summary>
        public Matrix<double> SupportVectors
        {
            get { return this.baseSvc.SupportVectors; }
        }

        /// <summary>
        /// [n_class * (n_class-1) / 2] Constants in decision function.
        /// </summary>
        public Vector<double> Intercept
        {
            get { return this.baseSvc.Intercept; }
        }

        public TLabel[] Classes
        {
            get { return this.baseSvc.Classes; }
        }

        /// <summary>
        /// [n_class-1, n_features]
        /// Weights asigned to the features (coefficients in the primal
        /// problem). This is only available in the case of linear kernel.
        ///
        // `coef_` is readonly property derived from `dual_coef_` and
        /// `support_vectors_`
        /// </summary>
        public Matrix<double> Coef
        {
            get { return this.baseSvc.Coef; }
        }

        public void Fit(Matrix<double> x, TLabel[] y)
        {
            this.baseSvc.Fit(x, y);
        }

        public TLabel[] Predict(Matrix<double> x)
        {
            return this.baseSvc.Predict(x);
        }

        public Matrix<double> PredictProba(Matrix<double> x)
        {
            return this.baseSvc.predict_proba(x);
        }
    }
}
