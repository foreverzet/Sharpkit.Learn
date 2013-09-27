// -----------------------------------------------------------------------
// <copyright file="LibSvmKernel.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Svm
{
    /// <summary>
    /// Libsvm kernel types.
    /// </summary>
    internal enum LibSvmKernel
    {
        Linear = 0,
        Poly = 1,
        Rbf = 2,
        Sigmoid = 3,
        Precomputed = 4
    }
}