// -----------------------------------------------------------------------
// <copyright file="lsvm.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using MathNet.Numerics.LinearAlgebra.Generic;

namespace Sharpkit.Learn.Svm
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public class lsvm
    {
        public static Matrix<double> predict(np.ndarray[np.float64_t, ndim=2, mode='c'] X,
            np.ndarray[np.int32_t, ndim=1, mode='c'] support,
            np.ndarray[np.float64_t, ndim=2, mode='c'] SV,
            np.ndarray[np.int32_t, ndim=1, mode='c'] nSV,
            np.ndarray[np.float64_t, ndim=2, mode='c'] sv_coef,
            np.ndarray[np.float64_t, ndim=1, mode='c'] intercept,
            np.ndarray[np.int32_t, ndim=1, mode='c'] label,
            np.ndarray[np.float64_t, ndim=1, mode='c'] probA=np.empty(0),
            np.ndarray[np.float64_t, ndim=1, mode='c'] probB=np.empty(0),
            int svm_type=0, str kernel='rbf', int degree=3,
            double gamma=0.1, double coef0=0., double tol=1e-3,
            double C=1., double nu=0.5, double epsilon=0.1,
            np.ndarray[np.float64_t, ndim=1, mode='c']
                class_weight=np.empty(0),
            np.ndarray[np.float64_t, ndim=1, mode='c']
                sample_weight=np.empty(0),
            int shrinking=0, int probability=0,
            double cache_size=100.,
            int max_iter=-1):
    """
    Predict target values of X given a model (low-level method)

    Parameters
    ----------
    X: array-like, dtype=float, size=[n_samples, n_features]

    svm_type : {0, 1, 2, 3, 4}
        Type of SVM: C SVC, nu SVC, one class, epsilon SVR, nu SVR

    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}
        Kernel to use in the model: linear, polynomial, RBF, sigmoid
        or precomputed.

    degree : int
        Degree of the polynomial kernel (only relevant if kernel is
        set to polynomial)

    gamma : float
        Gamma parameter in RBF kernel (only relevant if kernel is set
        to RBF)

    coef0 : float
        Independent parameter in poly/sigmoid kernel.

    eps : float
        Stopping criteria.

    C : float
        C parameter in C-Support Vector Classification


    Returns
    -------
    dec_values : array
        predicted values.


    TODO: probably there's no point in setting some parameters, like
    cache_size or weights.
    """
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] dec_values
    cdef svm_parameter param
    cdef svm_model *model
    cdef int rv

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)
    kernel_index = LIBSVM_KERNEL_TYPES.index(kernel)
    set_parameter(&param, svm_type, kernel_index, degree, gamma, coef0,
                          nu, cache_size, C, tol, epsilon, shrinking,
                          probability, <int> class_weight.shape[0],
                          class_weight_label.data, class_weight.data,
                          max_iter)

    model = set_model(&param, <int> nSV.shape[0], SV.data, SV.shape,
                      support.data, support.shape, sv_coef.strides,
                      sv_coef.data, intercept.data, nSV.data,
                      label.data, probA.data, probB.data)

    #TODO: use check_model
    dec_values = np.empty(X.shape[0])
    with nogil:
        rv = copy_predict(X.data, model, X.shape, dec_values.data)
    if rv < 0:
        raise MemoryError("We've run out of memory")
    free_model(model)
    return dec_values

    }
}
