#include <filesystem>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

class XorShift {
  public:
	XorShift(uint32_t seed = 123456789) : x(seed) {}

	uint32_t next() {
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		return x;
	}

	uint32_t next(uint32_t n) { return next() % n; }

	float nextFloat() {
		return static_cast<float>(next()) / static_cast<float>(UINT32_MAX);
	}

	float nextFloat(float min, float max) {
		return min + (max - min) * nextFloat();
	}

  private:
	uint32_t x;
};

std::map<std::string, std::string> argParse(int argc, char** argv) {
	std::map<std::string, std::string> args;
	std::string currentKey;
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg.starts_with("-")) {
			currentKey = arg;
			args[currentKey] = "";
		} else if (!currentKey.empty()) {
			args[currentKey] = arg;
		}
	}
	return args;
}

struct Shape {
	cv::Point2f center;
	float radius;
	cv::Vec3f color;
};

Shape shiftShape(const Shape& shape, float shiftX, float shiftY,
				 float shiftRadius) {
	Shape newShape;
	newShape.center.x = shape.center.x + shiftX;
	newShape.center.y = shape.center.y + shiftY;
	newShape.radius = shape.radius + shiftRadius;
	newShape.color = shape.color;
	return newShape;
}

struct CirclePolicy {
	static std::optional<cv::Vec2i> calcRangeX(const Shape& shape, const int y,
											   const int imgWidth) {
		const float yFloat = static_cast<float>(y);
		const float dy = yFloat - shape.center.y;
		const float radiusSq = shape.radius * shape.radius;
		const float distSq = dy * dy;
		if (distSq > radiusSq) {
			return std::nullopt;
		}
		const float dx = std::sqrt(radiusSq - distSq);
		int xStart = static_cast<int>(std::ceil(shape.center.x - dx));
		int xEnd = static_cast<int>(std::floor(shape.center.x + dx));
		xStart = std::max(0, xStart);
		xEnd = std::min(imgWidth - 1, xEnd);
		return cv::Vec2i(xStart, xEnd);
	}
};

struct SquarePolicy {
	static std::optional<cv::Vec2i> calcRangeX(const Shape& shape, const int y,
											   const int imgWidth) {
		const float yFloat = static_cast<float>(y);
		const float dy = std::abs(yFloat - shape.center.y);
		const float halfSide = shape.radius / std::sqrt(2.0f);
		if (dy > halfSide) {
			return std::nullopt;
		}
		int xStart = static_cast<int>(std::ceil(shape.center.x - halfSide));
		int xEnd = static_cast<int>(std::floor(shape.center.x + halfSide));
		xStart = std::max(0, xStart);
		xEnd = std::min(imgWidth - 1, xEnd);
		return cv::Vec2i(xStart, xEnd);
	}
};

struct DiamondPolicy {
	static std::optional<cv::Vec2i> calcRangeX(const Shape& shape, const int y,
											   const int imgWidth) {
		const float yFloat = static_cast<float>(y);
		const float dy = std::abs(yFloat - shape.center.y);
		if (dy > shape.radius) {
			return std::nullopt;
		}
		const float dx = shape.radius - dy;
		int xStart = static_cast<int>(std::ceil(shape.center.x - dx));
		int xEnd = static_cast<int>(std::floor(shape.center.x + dx));
		xStart = std::max(0, xStart);
		xEnd = std::min(imgWidth - 1, xEnd);
		return cv::Vec2i(xStart, xEnd);
	}
};

struct TriangleUpPolicy {
	static std::optional<cv::Vec2i> calcRangeX(const Shape& shape, const int y,
											   const int imgWidth) {
		const float yFloat = static_cast<float>(y);
		const float dy = yFloat - shape.center.y;
		if (dy < -shape.radius || dy > shape.radius * 0.5f) {
			return std::nullopt;
		}
		const float dx = (dy + shape.radius) / std::sqrt(3.0f);
		int xStart = static_cast<int>(std::ceil(shape.center.x - dx));
		int xEnd = static_cast<int>(std::floor(shape.center.x + dx));
		xStart = std::max(0, xStart);
		xEnd = std::min(imgWidth - 1, xEnd);
		return cv::Vec2i(xStart, xEnd);
	}
};

struct TriangleDownPolicy {
	static std::optional<cv::Vec2i> calcRangeX(const Shape& shape, const int y,
											   const int imgWidth) {
		const float yFloat = static_cast<float>(y);
		const float dy = yFloat - shape.center.y;
		if (dy > shape.radius || dy < -shape.radius * 0.5f) {
			return std::nullopt;
		}
		const float dx = (shape.radius - dy) / std::sqrt(3.0f);
		int xStart = static_cast<int>(std::ceil(shape.center.x - dx));
		int xEnd = static_cast<int>(std::floor(shape.center.x + dx));
		xStart = std::max(0, xStart);
		xEnd = std::min(imgWidth - 1, xEnd);
		return cv::Vec2i(xStart, xEnd);
	}
};

struct HexagonPolicy {
	static std::optional<cv::Vec2i> calcRangeX(const Shape& shape, const int y,
											   const int imgWidth) {
		const float yFloat = static_cast<float>(y);
		const float dy = std::abs(yFloat - shape.center.y);
		if (dy > shape.radius) {
			return std::nullopt;
		}
		float dx = 0.0f;
		if (dy > shape.radius * 0.5f) {
			dx = (shape.radius - dy) * std::sqrt(3.0f);
		} else {
			dx = shape.radius * std::sqrt(3.0f) * 0.5f;
		}
		int xStart = static_cast<int>(std::ceil(shape.center.x - dx));
		int xEnd = static_cast<int>(std::floor(shape.center.x + dx));
		xStart = std::max(0, xStart);
		xEnd = std::min(imgWidth - 1, xEnd);
		return cv::Vec2i(xStart, xEnd);
	}
};

float calcPixelLoss(const cv::Vec3f& accum, const int count,
					const cv::Vec3f& imageColor) {
	if (count == 0) {
		const int EMPTY_PENALTY = 20;
		return EMPTY_PENALTY;
	}
	const cv::Vec3f currentColor = accum / static_cast<float>(count);
	const cv::Vec3f diff = imageColor - currentColor;
	return diff.dot(diff);
}

float calcPixelLossDeltaAdd(const int x, const int y, const cv::Vec3f& color,
							const cv::Mat& image,
							const cv::Mat& canvasColorsAccum,
							const cv::Mat& canvasCounts) {
	const cv::Vec3f& imageColor = image.at<cv::Vec3f>(y, x);
	const cv::Vec3f& currentAccum = canvasColorsAccum.at<cv::Vec3f>(y, x);
	const int currentCount = canvasCounts.at<int>(y, x);
	const float currentLoss =
		calcPixelLoss(currentAccum, currentCount, imageColor);
	const float newLoss =
		calcPixelLoss(currentAccum + color, currentCount + 1, imageColor);
	return newLoss - currentLoss;
}

float calcPixelLossDeltaRemove(const int x, const int y, const cv::Vec3f& color,
							   const cv::Mat& image,
							   const cv::Mat& canvasColorsAccum,
							   const cv::Mat& canvasCounts) {
	const cv::Vec3f& imageColor = image.at<cv::Vec3f>(y, x);
	const cv::Vec3f& currentAccum = canvasColorsAccum.at<cv::Vec3f>(y, x);
	const int currentCount = canvasCounts.at<int>(y, x);
	const float currentLoss =
		calcPixelLoss(currentAccum, currentCount, imageColor);
	const float newLoss =
		calcPixelLoss(currentAccum - color, currentCount - 1, imageColor);
	return newLoss - currentLoss;
}

float calcLossDeltaFromRanges(const std::optional<cv::Vec2i>& currentXRangeOpt,
							  const std::optional<cv::Vec2i>& newXRangeOpt,
							  const int y, const cv::Vec3f& color,
							  const cv::Mat& image,
							  const cv::Mat& canvasColorsAccum,
							  const cv::Mat& canvasCounts) {
	float lossDelta = 0.0f;
	if (currentXRangeOpt.has_value()) {
		const cv::Vec2i currentXRange = currentXRangeOpt.value();
		if (newXRangeOpt.has_value()) {
			const cv::Vec2i newXRange = newXRangeOpt.value();
			if (newXRange[1] < currentXRange[0] ||
				currentXRange[1] < newXRange[0]) {
				// non-overlapping
				for (int x = currentXRange[0]; x <= currentXRange[1]; ++x) {
					lossDelta += calcPixelLossDeltaRemove(
						x, y, color, image, canvasColorsAccum, canvasCounts);
				}
				for (int x = newXRange[0]; x <= newXRange[1]; ++x) {
					lossDelta += calcPixelLossDeltaAdd(
						x, y, color, image, canvasColorsAccum, canvasCounts);
				}
			} else if (newXRange[0] <= currentXRange[0] &&
					   currentXRange[1] <= newXRange[1]) {
				// current is inside new
				for (int x = newXRange[0]; x < currentXRange[0]; ++x) {
					lossDelta += calcPixelLossDeltaAdd(
						x, y, color, image, canvasColorsAccum, canvasCounts);
				}
				for (int x = currentXRange[1] + 1; x <= newXRange[1]; ++x) {
					lossDelta += calcPixelLossDeltaAdd(
						x, y, color, image, canvasColorsAccum, canvasCounts);
				}
			} else if (currentXRange[0] <= newXRange[0] &&
					   newXRange[1] <= currentXRange[1]) {
				// new is inside current
				for (int x = currentXRange[0]; x < newXRange[0]; ++x) {
					lossDelta += calcPixelLossDeltaRemove(
						x, y, color, image, canvasColorsAccum, canvasCounts);
				}
				for (int x = newXRange[1] + 1; x <= currentXRange[1]; ++x) {
					lossDelta += calcPixelLossDeltaRemove(
						x, y, color, image, canvasColorsAccum, canvasCounts);
				}
			} else if (currentXRange[0] <= newXRange[0]) {
				// partial overlap, current left
				for (int x = currentXRange[0]; x < newXRange[0]; ++x) {
					lossDelta += calcPixelLossDeltaRemove(
						x, y, color, image, canvasColorsAccum, canvasCounts);
				}
				for (int x = currentXRange[1] + 1; x <= newXRange[1]; ++x) {
					lossDelta += calcPixelLossDeltaAdd(
						x, y, color, image, canvasColorsAccum, canvasCounts);
				}
			} else {
				// partial overlap, new left
				for (int x = newXRange[0]; x < currentXRange[0]; ++x) {
					lossDelta += calcPixelLossDeltaAdd(
						x, y, color, image, canvasColorsAccum, canvasCounts);
				}
				for (int x = newXRange[1] + 1; x <= currentXRange[1]; ++x) {
					lossDelta += calcPixelLossDeltaRemove(
						x, y, color, image, canvasColorsAccum, canvasCounts);
				}
			}
		} else {
			for (int x = currentXRange[0]; x <= currentXRange[1]; ++x) {
				lossDelta += calcPixelLossDeltaRemove(
					x, y, color, image, canvasColorsAccum, canvasCounts);
			}
		}
	} else {
		if (newXRangeOpt.has_value()) {
			const cv::Vec2i newXRange = newXRangeOpt.value();
			for (int x = newXRange[0]; x <= newXRange[1]; ++x) {
				lossDelta += calcPixelLossDeltaAdd(
					x, y, color, image, canvasColorsAccum, canvasCounts);
			}
		} else {
			// do nothing
		}
	}
	return lossDelta;
}

template <typename ShapePolicy>
cv::Mat renderShapes(const std::vector<Shape>& shapes,
					 const cv::Size canvasSize, const float scale = 1.0f) {
	cv::Mat canvas = cv::Mat::zeros(canvasSize, CV_32FC3);
	cv::Mat canvasColorsAccum = cv::Mat::zeros(canvasSize, CV_32FC3);
	cv::Mat canvasCounts = cv::Mat::zeros(canvasSize, CV_32SC1);
	for (auto shape : shapes) {
		shape.center.x *= scale;
		shape.center.y *= scale;
		shape.radius += 0.5f;
		shape.radius *= scale;
		const int yStart = std::max(
			0, static_cast<int>(std::ceil(shape.center.y - shape.radius)));
		const int yEnd = std::min(
			canvas.rows - 1,
			static_cast<int>(std::floor(shape.center.y + shape.radius)));
		for (int y = yStart; y <= yEnd; ++y) {
			const auto xRangeOpt =
				ShapePolicy::calcRangeX(shape, y, canvas.cols);
			if (!xRangeOpt.has_value()) {
				continue;
			}
			const cv::Vec2i xRange = xRangeOpt.value();
			canvasColorsAccum.at<cv::Vec3f>(y, xRange[0]) += shape.color;
			canvasCounts.at<int>(y, xRange[0]) += 1;
			if (xRange[1] != canvas.cols - 1) {
				canvasColorsAccum.at<cv::Vec3f>(y, xRange[1] + 1) -=
					shape.color;
				canvasCounts.at<int>(y, xRange[1] + 1) -= 1;
			}
		}
	}
	for (int y = 0; y < canvas.rows; ++y) {
		for (int x = 1; x < canvas.cols; ++x) {
			canvasColorsAccum.at<cv::Vec3f>(y, x) +=
				canvasColorsAccum.at<cv::Vec3f>(y, x - 1);
			canvasCounts.at<int>(y, x) += canvasCounts.at<int>(y, x - 1);
		}
	}

	for (int y = 0; y < canvas.rows; ++y) {
		for (int x = 0; x < canvas.cols; ++x) {
			const cv::Vec3f colorAccum = canvasColorsAccum.at<cv::Vec3f>(y, x);
			const int count = canvasCounts.at<int>(y, x);
			if (count > 0) {
				canvas.at<cv::Vec3f>(y, x) =
					colorAccum / static_cast<float>(count);
			}
		}
	}
	cv::Mat outputImage(canvas.rows, canvas.cols, CV_8UC3);
	for (int y = 0; y < canvas.rows; ++y) {
		for (int x = 0; x < canvas.cols; ++x) {
			const cv::Vec3f& color = canvas.at<cv::Vec3f>(y, x);
			outputImage.at<cv::Vec3b>(y, x) = cv::Vec3b(
				static_cast<uchar>(std::clamp(color[0] * 255.0f, 0.0f, 255.0f)),
				static_cast<uchar>(std::clamp(color[1] * 255.0f, 0.0f, 255.0f)),
				static_cast<uchar>(
					std::clamp(color[2] * 255.0f, 0.0f, 255.0f)));
		}
	}
	return outputImage;
}

template <typename ShapePolicy>
void runSplatting(const std::string& inputPath, const std::string& outputPath,
				  const int numberOfIterations, const int numberOfShapes,
				  const float initMaxRadius, const uint32_t seed) {
	XorShift rng(seed);

	cv::Mat originalImage = cv::imread(inputPath);
	if (originalImage.empty()) {
		std::cerr << "Failed to load image at " << inputPath << std::endl;
		return;
	}

	cv::Mat image;
	originalImage.convertTo(image, CV_32FC3, 1.0 / 255.0);
	const float scaleFactor =
		200.0f / static_cast<float>(std::max(image.cols, image.rows));
	const int newWidth = static_cast<int>(std::round(image.cols * scaleFactor));
	const int newHeight =
		static_cast<int>(std::round(image.rows * scaleFactor));
	cv::resize(image, image, cv::Size(newWidth, newHeight));

	std::vector<Shape> shapes(numberOfShapes);
	for (auto& shape : shapes) {
		shape.center = cv::Point2f(
			rng.nextFloat(0.0f, static_cast<float>(image.cols - 1)),
			rng.nextFloat(0.0f, static_cast<float>(image.rows - 1)));
		shape.radius =
			std::exp(rng.nextFloat(std::log(1.0f), std::log(initMaxRadius)));
		shape.color = image.at<cv::Vec3f>(static_cast<int>(shape.center.y),
										  static_cast<int>(shape.center.x));
	}
	for (int iter = 0; iter < numberOfIterations; ++iter) {
		if (iter % 10 == 0) {
			std::cout << "Iteration " << iter << std::endl;
		}

		cv::Mat canvasColorsAccum = cv::Mat::zeros(image.size(), CV_32FC3);
		cv::Mat canvasCounts = cv::Mat::zeros(image.size(), CV_32SC1);
		for (const auto& shape : shapes) {
			const int yStart = std::max(
				0, static_cast<int>(std::ceil(shape.center.y - shape.radius)));
			const int yEnd = std::min(
				image.rows - 1,
				static_cast<int>(std::floor(shape.center.y + shape.radius)));
			for (int y = yStart; y <= yEnd; ++y) {
				const auto xRangeOpt =
					ShapePolicy::calcRangeX(shape, y, image.cols);
				if (!xRangeOpt.has_value()) {
					continue;
				}
				const cv::Vec2i xRange = xRangeOpt.value();
				canvasColorsAccum.at<cv::Vec3f>(y, xRange[0]) += shape.color;
				canvasCounts.at<int>(y, xRange[0]) += 1;
				if (xRange[1] != image.cols - 1) {
					canvasColorsAccum.at<cv::Vec3f>(y, xRange[1] + 1) -=
						shape.color;
					canvasCounts.at<int>(y, xRange[1] + 1) -= 1;
				}
			}
		}
		for (int y = 0; y < image.rows; ++y) {
			for (int x = 1; x < image.cols; ++x) {
				canvasColorsAccum.at<cv::Vec3f>(y, x) +=
					canvasColorsAccum.at<cv::Vec3f>(y, x - 1);
				canvasCounts.at<int>(y, x) += canvasCounts.at<int>(y, x - 1);
			}
		}
		for (int i = 0; i < numberOfShapes; ++i) {
			cv::Vec3f dLdcolor = cv::Vec3f(0.0f, 0.0f, 0.0f);
			float dLdx = 0.0f;
			float dLdy = 0.0f;
			float dLdr = 0.0f;

			const auto& shape = shapes[i];
			const auto shapeShiftedX = shiftShape(shape, 1.0f, 0.0f, 0.0f);
			const auto shapeShiftedY = shiftShape(shape, 0.0f, 1.0f, 0.0f);
			const auto shapeShiftedRadius = shiftShape(shape, 0.0f, 0.0f, 1.0f);
			const int yStart = std::max(
				0,
				static_cast<int>(std::ceil(shape.center.y - shape.radius)) - 1);
			const int yEnd = std::min(
				image.rows - 1,
				static_cast<int>(std::floor(shape.center.y + shape.radius)) +
					1);
			for (int y = yStart; y <= yEnd; ++y) {
				const auto xRangeOpt =
					ShapePolicy::calcRangeX(shape, y, image.cols);
				const auto xRangeShiftedXOpt =
					ShapePolicy::calcRangeX(shapeShiftedX, y, image.cols);
				const auto xRangeShiftedYOpt =
					ShapePolicy::calcRangeX(shapeShiftedY, y, image.cols);
				const auto xRangeShiftedRadiusOpt =
					ShapePolicy::calcRangeX(shapeShiftedRadius, y, image.cols);

				dLdx += calcLossDeltaFromRanges(
					xRangeOpt, xRangeShiftedXOpt, y, shape.color, image,
					canvasColorsAccum, canvasCounts);
				dLdy += calcLossDeltaFromRanges(
					xRangeOpt, xRangeShiftedYOpt, y, shape.color, image,
					canvasColorsAccum, canvasCounts);
				dLdr += calcLossDeltaFromRanges(
					xRangeOpt, xRangeShiftedRadiusOpt, y, shape.color, image,
					canvasColorsAccum, canvasCounts);
				if (!xRangeOpt.has_value()) {
					continue;
				}
				if (iter % 10 == 0) {
					const cv::Vec2i xRange = xRangeOpt.value();
					for (int x = xRange[0]; x <= xRange[1]; ++x) {
						const cv::Vec3f& imageColor = image.at<cv::Vec3f>(y, x);
						const cv::Vec3f& currentColorAccum =
							canvasColorsAccum.at<cv::Vec3f>(y, x);
						const int currentCount = canvasCounts.at<int>(y, x);
						const cv::Vec3f currentColor =
							currentCount != 0
								? currentColorAccum /
									  static_cast<float>(currentCount)
								: cv::Vec3f(0.0f, 0.0f, 0.0f);
						const cv::Vec3f diff = currentColor - imageColor;
						for (int c = 0; c < 3; ++c) {
							dLdcolor[c] += diff[c] * std::abs(diff[c]) /
										   static_cast<float>(currentCount);
						}
					}
				}
			}

			// Update circle parameters
			const float learningRateColor = 1.0f;

			const float radiusSq = shape.radius * shape.radius;
			shapes[i].color -= learningRateColor * dLdcolor / radiusSq;
			for (int c = 0; c < 3; ++c) {
				shapes[i].color[c] = std::clamp(shapes[i].color[c], 0.0f, 1.0f);
			}

			const float learningRatePosition = 1.0f;
			shapes[i].center.x -=
				learningRatePosition * dLdx / shapes[i].radius;
			shapes[i].center.y -=
				learningRatePosition * dLdy / shapes[i].radius;
			shapes[i].center.x = std::clamp(shapes[i].center.x, 0.0f,
											static_cast<float>(image.cols - 1));
			shapes[i].center.y = std::clamp(shapes[i].center.y, 0.0f,
											static_cast<float>(image.rows - 1));
			const float learningRateRadius = 0.01f;
			shapes[i].radius -= learningRateRadius * dLdr / shapes[i].radius;
			const float maxRadius = std::min(image.cols, image.rows);
			shapes[i].radius = std::clamp(shapes[i].radius, 1.0f, maxRadius);
		}
	}

	cv::Mat finalImage = renderShapes<ShapePolicy>(shapes, originalImage.size(),
												   1.0f / scaleFactor);
	cv::imwrite(outputPath, finalImage);
}

int main(int argc, char** argv) {
	const auto args = argParse(argc, argv);

	std::string inputPath, outputPath, mode, numberOfIterationsStr,
		numberOfShapesStr, initMaxRadiusStr, seedStr;
	if (args.find("-i") != args.end()) {
		inputPath = args.at("-i");
	} else if (args.find("--input") != args.end()) {
		inputPath = args.at("--input");
	} else {
		std::cerr << "Input image path not provided. Use -i or --input to "
					 "specify the path."
				  << std::endl;
		return -1;
	}

	if (args.find("-o") != args.end()) {
		outputPath = args.at("-o");
	} else if (args.find("--output") != args.end()) {
		outputPath = args.at("--output");
	} else {
		std::cerr << "Output image path not provided. Use -o or --output to "
					 "specify the path."
				  << std::endl;
		return -1;
	}

	if (args.find("-m") != args.end()) {
		mode = args.at("-m");
	} else if (args.find("--mode") != args.end()) {
		mode = args.at("--mode");
	} else {
		mode = "circle";  // default mode
	}

	if (args.find("-t") != args.end()) {
		numberOfIterationsStr = args.at("-t");
	} else if (args.find("--iter") != args.end()) {
		numberOfIterationsStr = args.at("--iter");
	} else {
		numberOfIterationsStr = "1000";	 // default value
	}

	if (args.find("-s") != args.end()) {
		numberOfShapesStr = args.at("-s");
	} else if (args.find("--shapes") != args.end()) {
		numberOfShapesStr = args.at("--shapes");
	} else {
		numberOfShapesStr = "400";	// default value
	}

	if (args.find("-r") != args.end()) {
		initMaxRadiusStr = args.at("-r");
	} else if (args.find("--max-radius") != args.end()) {
		initMaxRadiusStr = args.at("--max-radius");
	} else {
		initMaxRadiusStr = "20";  // default value
	}

	if (args.find("-s") != args.end()) {
		seedStr = args.at("-s");
	} else if (args.find("--seed") != args.end()) {
		seedStr = args.at("--seed");
	} else {
		seedStr = "123456789";	// default value
	}

	const int numberOfIterations = std::stoi(numberOfIterationsStr);
	const int numberOfShapes = std::stoi(numberOfShapesStr);
	const float initMaxRadius = std::max(1.0f, std::stof(initMaxRadiusStr));

	const uint32_t seed = static_cast<uint32_t>(std::stoul(seedStr));
	if (seed == 0) {
		std::cerr << "Seed must be a positive integer." << std::endl;
		return -1;
	}
	if (mode == "circle") {
		runSplatting<CirclePolicy>(inputPath, outputPath, numberOfIterations,
								   numberOfShapes, initMaxRadius, seed);
	}
	if (mode == "square") {
		runSplatting<SquarePolicy>(inputPath, outputPath, numberOfIterations,
								   numberOfShapes, initMaxRadius, seed);
	} else if (mode == "diamond") {
		runSplatting<DiamondPolicy>(inputPath, outputPath, numberOfIterations,
									numberOfShapes, initMaxRadius, seed);
	} else if (mode == "triangle-up") {
		runSplatting<TriangleUpPolicy>(inputPath, outputPath,
									   numberOfIterations, numberOfShapes,
									   initMaxRadius, seed);
	} else if (mode == "triangle-down") {
		runSplatting<TriangleDownPolicy>(inputPath, outputPath,
										 numberOfIterations, numberOfShapes,
										 initMaxRadius, seed);
	} else if (mode == "hexagon") {
		runSplatting<HexagonPolicy>(inputPath, outputPath, numberOfIterations,
									numberOfShapes, initMaxRadius, seed);
	} else {
		std::cerr << "Unsupported mode: " << mode << std::endl;
		return -1;
	}
}