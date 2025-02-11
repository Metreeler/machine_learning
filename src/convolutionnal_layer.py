import numpy as np
import cv2 as cv
import json

from layer import Layer

class ConvolutionnalLayer(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, learning_rate):
        super().__init__()
        
        
        self.kernel_size = kernel_size
        self.kernels = np.random.rand(out_channels, in_channels, kernel_size, kernel_size) * 2 - 1
        
        
        for i in range(out_channels):
            for j in range(in_channels):
                self.kernels[i, j] = (self.kernels[i, j] - np.min(self.kernels[i, j])) / (np.max(self.kernels[i, j]) - np.min(self.kernels[i, j])) - 0.5
        
        self.stride = stride
        self.formatted_inputs = np.array([])
        
        self.learning_rate = learning_rate
    
    def forward_propagation(self, input_vector):
        
        self.input_vector = input_vector
    
        formatted_inputs = np.zeros((self.kernel_size, 
                                     self.kernel_size, 
                                     1, 
                                     input_vector.shape[0], 
                                     self.kernels.shape[1], 
                                     int((input_vector.shape[2] - self.kernel_size) / self.stride + 1), 
                                     int((input_vector.shape[3] - self.kernel_size) / self.stride + 1)))
        
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                formatted_inputs[i, j, :] = input_vector[:, 
                                                         :, 
                                                         i:(i + input_vector.shape[2] - self.kernel_size + 1):self.stride, 
                                                         j:(j + input_vector.shape[3] - self.kernel_size + 1):self.stride]
        
        formatted_inputs = np.transpose(formatted_inputs, (3, 2, 4, 0, 1, 5, 6))
        
        self.formatted_inputs = formatted_inputs
        
        shaped_kernels = np.reshape(self.kernels, (1, self.kernels.shape[0], self.kernels.shape[1], self.kernels.shape[2], self.kernels.shape[3], 1, 1))
        
        out = np.multiply(formatted_inputs, shaped_kernels)
        
        out = np.sum(np.sum((np.sum(out, axis=2)), axis=2), axis=2)
        
        out /= np.reshape(np.max(np.abs(out), axis=(2, 3)), (out.shape[0], out.shape[1], 1, 1))
        
        return out
    
    def backward_propagation(self, partial_derivative):
        if np.max(np.abs(partial_derivative)) > 1:
            partial_derivative /= np.reshape(np.max(np.abs(partial_derivative), axis=(2, 3)), (partial_derivative.shape[0], partial_derivative.shape[1], 1, 1))
        
        formatted_partial_derivative = np.reshape(partial_derivative, (partial_derivative.shape[0], partial_derivative.shape[1], 1, 1, 1, partial_derivative.shape[2], partial_derivative.shape[3]))
        
        kernels_modifiers = np.multiply(formatted_partial_derivative, self.formatted_inputs)
        kernels_modifiers = np.sum(np.sum(kernels_modifiers, axis=5), axis=5)
        kernels_modifiers = np.mean(kernels_modifiers, axis=0)
        
        self.kernels -= kernels_modifiers * self.learning_rate
        
        for i in range(self.kernels.shape[0]):
            for j in range(self.kernels.shape[1]):
                if np.max(np.abs(self.kernels[i, j])) > 1:
                    self.kernels[i, j] /= np.max(np.abs(self.kernels[i, j]))
        
        partial_derivative = np.reshape(partial_derivative, (partial_derivative.shape[0], partial_derivative.shape[1], 1, 1, 1, partial_derivative.shape[2], partial_derivative.shape[3]))
        
        shaped_kernels = np.reshape(self.kernels, (1, self.kernels.shape[0], self.kernels.shape[1], self.kernels.shape[2], self.kernels.shape[3], 1, 1))
        
        shaped_out = np.multiply(partial_derivative, shaped_kernels)
        
        shaped_out = np.mean(shaped_out, axis=1)
        
        out = np.zeros(self.input_vector.shape)
        out_divider = np.zeros((out.shape[2], out.shape[3]))
        
        for i in range(shaped_out.shape[2]):
            for j in range(shaped_out.shape[3]):
                out[:, 
                    :, 
                    i:(i + self.input_vector.shape[2] - self.kernel_size + 1):self.stride, 
                    j:(j + self.input_vector.shape[3] - self.kernel_size + 1):self.stride] += shaped_out[:, :, i, j, :, :]
                    
                out_divider[i:(i + self.input_vector.shape[2] - self.kernel_size + 1):self.stride,
                            j:(j + self.input_vector.shape[3] - self.kernel_size + 1):self.stride] += 1
        
        out_divider = np.where(out_divider == 0, 1, out_divider)
        out = out / out_divider
        
        return out
    
    def render_kernels(self, kernel_spacing, filename=None):
        out = np.zeros(((self.kernel_size + kernel_spacing) * self.kernels.shape[0] + kernel_spacing, 
                        (self.kernel_size + kernel_spacing) * self.kernels.shape[1] + kernel_spacing, 3)) + 255
        
        for i in range(self.kernels.shape[0]):
            for j in range(self.kernels.shape[1]):
                out[((self.kernel_size + kernel_spacing) * i + kernel_spacing):((self.kernel_size + kernel_spacing) * i + kernel_spacing + self.kernel_size), 
                    ((self.kernel_size + kernel_spacing) * j + kernel_spacing):((self.kernel_size + kernel_spacing) * j + kernel_spacing + self.kernel_size), :] = draw_green_red_values(self.kernels[i, j])
        if filename is not None:
            cv.imwrite(filename, out)
        return out


def draw_green_red_values(src):
    out = np.zeros((src.shape[0], src.shape[1], 3))
    max_value = np.max(np.absolute(src))
    
    out[:, :, 2] = np.where(src < 0, -255 * src / max_value, 0)
    out[:, :, 1] = np.where(src > 0, 255 * src / max_value, 0)
    
    return out


if __name__ == "__main__":
    img = cv.imread("data/img_test.png")
    img2 = cv.imread("data/img_test_2.png")
    
    # images = np.zeros((2, 3, 30, 30))
    # images[0] = np.transpose(img, (2, 0, 1))
    # images[1] = np.transpose(img2, (2, 0, 1))
    
    images = np.zeros((2, 1, 30, 30))
    images[0] = img[:, :, 0]
    images[1] = img2[:, :, 0]
    images = images / 255
    
    with open("data/convolution_result.json", "r") as f:
        y = np.array(json.load(f)["data"])
    
    cl = ConvolutionnalLayer(1, 1, 3, 1)

    cl2  = ConvolutionnalLayer(1, 1, 3, 1)
    
    # cl.kernels = np.array([[[[-1, -2, -1],
    #                                [0, 0, 0],
    #                                [1, 2, 1]],
    #                               [[-1, -2, -1],
    #                                [0, 0, 0],
    #                                [1, 2, 1]],
    #                               [[-1, -2, -1],
    #                                [0, 0, 0],
    #                                [1, 2, 1]]]]).astype(float)
    
    # cl.kernels[0, :, :, :] = cl.kernels[0, :, :, :] / 2
    
    # cl2.kernels = np.array([[[[-1, 0, 1],
    #                                [-2, 0, 2],
    #                                [-1, 0, 1]]]]).astype(float)
    
    # cl2.kernels[0, :, :, :] = cl2.kernels[0, :, :, :] / 2
    
    # y /= np.reshape(np.max(np.abs(y), axis=(2, 3)), (y.shape[0], y.shape[1], 1, 1))
    for i in range(1):
        print("Epoch :", i)
        cl.render_kernels(3, "data/kernels/0_" + str(i) + ".png")
        cl2.render_kernels(3, "data/kernels/1_" + str(i) + ".png")

        out = cl.forward_propagation(images)
        out = cl2.forward_propagation(out)
        
        loss = out - y
        
        loss = cl2.backward_propagation(loss, 0.01)
        loss = cl.backward_propagation(loss, 0.01)
    
    cl.render_kernels(3)
    cl2.render_kernels(3, "data/kernels2.png")
    
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            cv.imwrite("data/img_out/" + str(i) + "_" + str(j) + ".png", draw_green_red_values(out[i, j]))
    
    
    # out /= np.reshape(np.max(np.abs(out), axis=(2, 3)), (out.shape[0], out.shape[1], 1, 1))
    # for i in range(out.shape[0]):
    #     for j in range(out.shape[1]):
    #         print(np.max(out[i, j]))
    # print(np.max(np.abs(out), axis=(2, 3)))
    
    
    # with open("data/convolution_result.json", "w") as f:
    #     json.dump({"data":out.tolist()}, f)
    