// -----------------------------------------------------------------------
// <copyright file="IrisDatasetTest.cs" company="Sharpkit.Learn">
// # Copyright (c) 2007 David Cournapeau &lt;cournape@gmail.com>
//   2010 Fabian Pedregosa &lt;fabian.pedregosa@inria.fr>
//   2010 Olivier Grisel &lt;olivier.grisel@ensta.org>
//   2013 Sergey Zyuzin &lt;forever.zet@gmail.com>
// License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Test.Datasets
{
    using System;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Sharpkit.Learn.Datasets;

    /// <summary>
    /// Tests for <see cref="IrisDataset"/>.
    /// </summary>
    [TestClass]
    public class IrisDatasetTest
    {
        /// <summary>
        /// Tests loading iris dataset.
        /// </summary>
        [TestMethod]
        public void TestLoad()
        {
            var res = IrisDataset.Load();
            Assert.AreEqual(res.Data.Shape(), Tuple.Create(150, 4));
            Assert.AreEqual(res.Target.Length, 150);
        }
    }
}
