#include <filesystem>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>

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

struct Circle {
    cv::Point2f center;
    float radius;
    cv::Vec3f color;
};

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

cv::Mat renderCircles(const std::vector<Circle>& circles, cv::Size originalCanvasSize, float scale = 1.0f) {
    cv::Mat canvas = cv::Mat::zeros(originalCanvasSize, CV_32FC3);
    cv::Mat canvasColorsAccum = cv::Mat::zeros(originalCanvasSize, CV_32FC3);
    cv::Mat canvasCounts = cv::Mat::zeros(originalCanvasSize, CV_32SC1);
    for (auto circle : circles) {
        circle.center.x *= scale;
        circle.center.y *= scale;
        circle.radius *= scale;
        const int xStart = std::max(
            0, static_cast<int>(std::floor(circle.center.x - circle.radius)));
        const int xEnd = std::min(
            canvas.cols - 1,
            static_cast<int>(std::ceil(circle.center.x + circle.radius)));
        const int yStart = std::max(
            0, static_cast<int>(std::floor(circle.center.y - circle.radius)));
        const int yEnd = std::min(
            canvas.rows - 1,
            static_cast<int>(std::ceil(circle.center.y + circle.radius)));
        const float radiusSq = circle.radius * circle.radius;
        for (int y = yStart; y <= yEnd; ++y) {
            for (int x = xStart; x <= xEnd; ++x) {
                const float dx = static_cast<float>(x) - circle.center.x;
                const float dy = static_cast<float>(y) - circle.center.y;
                const float distSq = dx * dx + dy * dy;
                if (distSq <= radiusSq) {
                    canvasColorsAccum.at<cv::Vec3f>(y, x) += circle.color;
                    canvasCounts.at<int>(y, x) += 1;
                }
            }
        }
    }
    for (int y = 0; y < canvas.rows; ++y) {
        for (int x = 0; x < canvas.cols; ++x) {
            const cv::Vec3f colorAccum =
                canvasColorsAccum.at<cv::Vec3f>(y, x);
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
                static_cast<uchar>(std::clamp(color[2] * 255.0f, 0.0f, 255.0f)));
        }
    }
    return outputImage;
}

int main(int argc, char** argv) {
    const auto args = argParse(argc, argv);

    std::string inputPath, outputPath, numberOfIterationsStr,
        numberOfCirclesStr, seedStr;
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

    if (args.find("-t") != args.end()) {
        numberOfIterationsStr = args.at("-t");
    } else if (args.find("--iter") != args.end()) {
        numberOfIterationsStr = args.at("--iter");
    } else {
        numberOfIterationsStr = "1000";  // default value
    }

    if (args.find("-c") != args.end()) {
        numberOfCirclesStr = args.at("-c");
    } else if (args.find("--circles") != args.end()) {
        numberOfCirclesStr = args.at("--circles");
    } else {
        numberOfCirclesStr = "200";  // default value
    }

    if (args.find("-s") != args.end()) {
        seedStr = args.at("-s");
    } else if (args.find("--seed") != args.end()) {
        seedStr = args.at("--seed");
    } else {
        seedStr = "123456789";  // default value
    }

    const int numberOfIterations = std::stoi(numberOfIterationsStr);
    const int numberOfCircles = std::stoi(numberOfCirclesStr);
    const uint32_t seed = static_cast<uint32_t>(std::stoul(seedStr));
    if (seed == 0) {
        std::cerr << "Seed must be a positive integer." << std::endl;
        return -1;
    }
    XorShift rng(seed);

    cv::Mat originalImage = cv::imread(inputPath);
    if (originalImage.empty()) {
        std::cerr << "Failed to load image at " << inputPath << std::endl;
        return -1;
    }

    cv::Mat image;
    originalImage.convertTo(image, CV_32FC3, 1.0 / 255.0);
    const float scaleFactor =
        200.0f / static_cast<float>(std::max(image.cols, image.rows));
    const int newWidth =
        static_cast<int>(std::round(image.cols * scaleFactor));
    const int newHeight =
        static_cast<int>(std::round(image.rows * scaleFactor));
    cv::resize(image, image, cv::Size(newWidth, newHeight));

    std::vector<Circle> circles(numberOfCircles);
    for (auto& circle : circles) {
        circle.center = cv::Point2f(
            rng.nextFloat(0.0f, static_cast<float>(image.cols - 1)),
            rng.nextFloat(0.0f, static_cast<float>(image.rows - 1)));
        const float randomMaxRadius = std::min(image.cols, image.rows) / 5.0f;
        const float randomMinRadius = std::min(5.0f, randomMaxRadius);
        circle.radius = rng.nextFloat(randomMinRadius, randomMaxRadius);
        circle.color =
            cv::Vec3f(rng.nextFloat(0.0f, 1.0f), rng.nextFloat(0.0f, 1.0f),
                      rng.nextFloat(0.0f, 1.0f));
    }
    for (int iter = 0; iter < numberOfIterations; ++iter) {
        cv::Mat canvasColorsAccum = cv::Mat::zeros(image.size(), CV_32FC3);
        cv::Mat canvasCounts = cv::Mat::zeros(image.size(), CV_32SC1);
        for (const auto& circle : circles) {
            const int xStart = std::max(
                0,
                static_cast<int>(std::floor(circle.center.x - circle.radius)));
            const int xEnd = std::min(
                image.cols - 1,
                static_cast<int>(std::ceil(circle.center.x + circle.radius)));
            const int yStart = std::max(
                0,
                static_cast<int>(std::floor(circle.center.y - circle.radius)));
            const int yEnd = std::min(
                image.rows - 1,
                static_cast<int>(std::ceil(circle.center.y + circle.radius)));
            for (int y = yStart; y <= yEnd; ++y) {
                for (int x = xStart; x <= xEnd; ++x) {
                    const float dx = static_cast<float>(x) - circle.center.x;
                    const float dy = static_cast<float>(y) - circle.center.y;
                    const float distSq = dx * dx + dy * dy;
                    if (distSq <= circle.radius * circle.radius) {
                        canvasColorsAccum.at<cv::Vec3f>(y, x) += circle.color;
                        canvasCounts.at<int>(y, x) += 1;
                    }
                }
            }
        }

        if (iter % 10 == 0) {
            std::cout << "Iteration " << iter << std::endl;
            cv::Mat outputImage = renderCircles(circles, image.size());
            cv::imwrite(std::to_string(iter) + "_" + outputPath, outputImage);
        }

        for (int i = 0; i < numberOfCircles; ++i) {
            cv::Vec3f dLdcolor = cv::Vec3f(0.0f, 0.0f, 0.0f);

            const auto& circle = circles[i];
            const int xStart = std::max(
                0,
                static_cast<int>(std::floor(circle.center.x - circle.radius)) -
                    1);
            const int xEnd = std::min(
                image.cols - 1,
                static_cast<int>(std::ceil(circle.center.x + circle.radius)) +
                    1);
            const int yStart = std::max(
                0,
                static_cast<int>(std::floor(circle.center.y - circle.radius)) -
                    1);
            const int yEnd = std::min(
                image.rows - 1,
                static_cast<int>(std::ceil(circle.center.y + circle.radius)) +
                    1);
            const float radiusSq = circle.radius * circle.radius;

            for (int y = yStart; y <= yEnd; ++y) {
                for (int x = xStart; x <= xEnd; ++x) {
                    const float xFloat = static_cast<float>(x);
                    const float yFloat = static_cast<float>(y);
                    const float dx = xFloat - circle.center.x;
                    const float dy = yFloat - circle.center.y;

                    const float distSq = dx * dx + dy * dy;
                    const bool inCircle = distSq <= radiusSq;

                    const cv::Vec3f& imageColor = image.at<cv::Vec3f>(y, x);
                    const cv::Vec3f& currentColorAccum =
                        canvasColorsAccum.at<cv::Vec3f>(y, x);
                    const int currentCount = canvasCounts.at<int>(y, x);
                    const cv::Vec3f currentColor =
                        currentCount != 0 ? currentColorAccum /
                                                static_cast<float>(currentCount)
                                          : cv::Vec3f(-1.0f, -1.0f, -1.0f);
                    if (inCircle) {
                        const cv::Vec3f diff = currentColor - imageColor;
                        for (int c = 0; c < 3; ++c) {
                            dLdcolor[c] +=
                                diff[c] / static_cast<float>(currentCount);
                        }
                    }
                }
            }

            auto calcLossDelta = [&](float deltaX, float deltaY,
                                     float deltaRadius) -> float {
                float deltaLoss = 0.0f;

                const float radiusNew = circle.radius + deltaRadius;
                const float centerXNew = circle.center.x + deltaX;
                const float centerYNew = circle.center.y + deltaY;
                const float radiusNewSq = radiusNew * radiusNew;

                const float maxRadius = std::max(circle.radius, radiusNew);
                const int xStart = std::max(
                    0,
                    static_cast<int>(std::floor(centerXNew - maxRadius)) - 1);
                const int xEnd = std::min(
                    image.cols - 1,
                    static_cast<int>(std::ceil(centerXNew + maxRadius)) + 1);
                const int yStart = std::max(
                    0,
                    static_cast<int>(std::floor(centerYNew - maxRadius)) - 1);
                const int yEnd = std::min(
                    image.rows - 1,
                    static_cast<int>(std::ceil(centerYNew + maxRadius)) + 1);

                for (int y = yStart; y <= yEnd; ++y) {
                    for (int x = xStart; x <= xEnd; ++x) {
                        const float xFloat = static_cast<float>(x);
                        const float yFloat = static_cast<float>(y);
                        const float oldDx = xFloat - circle.center.x;
                        const float oldDy = yFloat - circle.center.y;
                        const float newDx = xFloat - centerXNew;
                        const float newDy = yFloat - centerYNew;

                        const bool inOld =
                            (oldDx * oldDx + oldDy * oldDy) <= radiusSq;
                        const bool inNew =
                            (newDx * newDx + newDy * newDy) <= radiusNewSq;
                        if (inOld == inNew) {
                            continue;
                        }

                        const cv::Vec3f& imageColor = image.at<cv::Vec3f>(y, x);
                        const cv::Vec3f& currentAccum =
                            canvasColorsAccum.at<cv::Vec3f>(y, x);
                        const int currentCount = canvasCounts.at<int>(y, x);
                        const cv::Vec3f& currentColor =
                            currentCount != 0
                                ? currentAccum /
                                      static_cast<float>(currentCount)
                                : cv::Vec3f(-1.0f, -1.0f, -1.0f);
                        float currentPixelLoss = 0.0f;
                        for (int c = 0; c < 3; ++c) {
                            currentPixelLoss +=
                                std::abs(currentColor[c] - imageColor[c]);
                        }

                        cv::Vec3f nextAccum;
                        int nextCount;

                        if (inOld && !inNew) {
                            nextAccum = currentAccum - circles[i].color;
                            nextCount = currentCount - 1;
                        } else {
                            nextAccum = currentAccum + circles[i].color;
                            nextCount = currentCount + 1;
                        }
                        const cv::Vec3f nextColor =
                            nextCount != 0
                                ? nextAccum / static_cast<float>(nextCount)
                                : cv::Vec3f(-1.0f, -1.0f, -1.0f);
                        float nextPixelLoss = 0.0f;
                        for (int c = 0; c < 3; ++c) {
                            nextPixelLoss +=
                                std::abs(nextColor[c] - imageColor[c]);
                        }
                        deltaLoss += nextPixelLoss - currentPixelLoss;
                    }
                }
                return deltaLoss;
            };

            // Update circle parameters
            const float learningRateColor = 1.0f;

            circles[i].color -= learningRateColor * dLdcolor / radiusSq;
            for (int c = 0; c < 3; ++c) {
                circles[i].color[c] =
                    std::clamp(circles[i].color[c], 0.0f, 1.0f);
            }

            const float dLdx = calcLossDelta(1.0f, 0.0f, 0.0f);
            const float dLdy = calcLossDelta(0.0f, 1.0f, 0.0f);
            const float dLdr = calcLossDelta(0.0f, 0.0f, 1.0f);

            const float learningRatePosition = 10.0f;
            circles[i].center.x -=
                learningRatePosition * dLdx / circles[i].radius;
            circles[i].center.y -=
                learningRatePosition * dLdy / circles[i].radius;
            circles[i].center.x = std::clamp(
                circles[i].center.x, 0.0f, static_cast<float>(image.cols - 1));
            circles[i].center.y = std::clamp(
                circles[i].center.y, 0.0f, static_cast<float>(image.rows - 1));
            const float learningRateRadius = 10.0f;
            circles[i].radius -= learningRateRadius * dLdr / circles[i].radius;
            const float maxRadius = std::min(image.cols, image.rows);
            circles[i].radius = std::clamp(circles[i].radius, 1.0f, maxRadius);
        }
    }

    cv::Mat finalImage = renderCircles(circles, originalImage.size(), 1.0f / scaleFactor);
    cv::imwrite(outputPath, finalImage);
}
