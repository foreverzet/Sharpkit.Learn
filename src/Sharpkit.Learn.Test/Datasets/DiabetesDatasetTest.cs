// -----------------------------------------------------------------------
// <copyright file="DiabetesDatasetTest.cs" company="Sharpkit.Learn">
//   Copyright (c) 2007 David Cournapeau &lt;cournape@gmail.com>
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
    /// Tests for <see cref="DiabetesDataset"/>.
    /// </summary>
    [TestClass]
    public class DiabetesDatasetTest
    {
        /// <summary>
        /// Tests loading diabetes dataset.
        /// </summary>
        [TestMethod]
        public void TestLoad()
        {
            var res = DiabetesDataset.Load();
            Assert.AreEqual(res.Data.Shape(), Tuple.Create(442, 10));
            Assert.AreEqual(res.Target.Count, 442);
        }
    }
}
