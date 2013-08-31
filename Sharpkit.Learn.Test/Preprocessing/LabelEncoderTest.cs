// -----------------------------------------------------------------------
// <copyright file="LabelEncoderTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Test.Preprocessing
{
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Sharpkit.Learn.Preprocessing;
    using System;
    using System.Linq;

    /// <summary>
    /// Tests <see cref="LabelEncoder{TLabel}"/> class.
    /// </summary>
    [TestClass]
    public class LabelEncoderTest
    {
        /// <summary>
        /// Test LabelEncoder's transform and inverse_transform methods.
        /// </summary>
        [TestMethod]
        public void TestLabelEncoder()
        {
            var le = new LabelEncoder<int>();
            le.Fit(new[] {1, 1, 4, 5, -1, 0});
            Assert.IsTrue(new[] {-1, 0, 1, 4, 5}.SequenceEqual(le.Classes));
            Assert.IsTrue(new []{1, 2, 3, 3, 4, 0, 0}.SequenceEqual(le.Transform(new []{0, 1, 4, 4, 5, -1, -1})));
            Assert.IsTrue(new[] {0, 1, 4, 4, 5, -1, -1}.SequenceEqual(le.InverseTransform(new[] {1, 2, 3, 3, 4, 0, 0})));
            try
            {
                le.Transform(new[] {0, 6});
                Assert.Fail("ArgumentException is expected");
            }
            catch (ArgumentException)
            {
            }
        }

        /// <summary>
        /// Tests FitTransform.
        /// </summary>
        [TestMethod]
        public void TestLabelEncoderFitTransform()
        {
            var le = new LabelEncoder<int>();
            var ret = le.FitTransform(new[] {1, 1, 4, 5, -1, 0});
            Assert.IsTrue(new[] {2, 2, 3, 4, 0, 1}.SequenceEqual(ret));

            var le1 = new LabelEncoder<string>();
            ret = le1.FitTransform(new[] {"paris", "paris", "tokyo", "amsterdam"});
            Assert.IsTrue(new[] {1, 1, 2, 0}.SequenceEqual(ret));
        }

        /// <summary>
        /// Test LabelEncoder's transform and inverse_transform methods with
        /// non-numeric labels
        /// </summary>
        [TestMethod]
        public void TestLabelEncoderStringLabels()
        {
            var le = new LabelEncoder<string>();
            le.Fit(new[] {"paris", "paris", "tokyo", "amsterdam"});
            Assert.IsTrue(new[] {"amsterdam", "paris", "tokyo"}.SequenceEqual(le.Classes));
            Assert.IsTrue(new[] {2, 2, 1}.SequenceEqual(le.Transform(new[] {"tokyo", "tokyo", "paris"})));
            Assert.IsTrue(new[] {"tokyo", "tokyo", "paris"}.SequenceEqual(le.InverseTransform(new[] {2, 2, 1})));
            
            try
            {
                le.Transform(new []{"london"});
                Assert.Fail("ArgumentException expected");
            }
            catch (Exception){}
        }

        /// <summary>
        /// Check that invalid arguments yield ArgumentException.
        /// </summary>
        [TestMethod]
        public void TestLabelEncoderErrors()
        {
            var le = new LabelEncoder<int>();
            try
            {
                le.Transform(new int[0]);
                Assert.Fail("ArgumentException expected");
            }
            catch (Exception){}

            try
            {
                le.InverseTransform(new int[0]);
                Assert.Fail("ArgumentException expected");
            }
            catch (Exception){}
        }
    }
}
