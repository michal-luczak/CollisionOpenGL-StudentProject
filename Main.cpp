#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb/stb_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

#include "Texture.h"
#include "shaderClass.h"
#include "VAO.h"
#include "VBO.h"
#include "EBO.h"
#include "Camera.h"

const unsigned int width = 1280;
const unsigned int height = 720;
const unsigned int stacks = 30;
const unsigned int sectors = 30;
float radius = 0.1f;
const float PI = 3.14159;
const float dt = 0.001;
float restitution = 0.8f; // Default restitution coefficient

const unsigned short spheresCount = 15;
int vertexIndex = 0;
int index = 0;

const glm::vec3 GRAVITY(0.0f, -9.81f, 0.0f); // Gravity vector

struct State {
    glm::vec3 position;
    glm::vec3 velocity;
};

struct Derivative {
    glm::vec3 dPosition;
    glm::vec3 dVelocity;
};

void generateVerticesAndIndices(GLfloat* vertices, GLuint* indices) {
    const float sectorStep = 2 * PI / sectors;
    const float stackStep = PI / stacks;

    for (int i = 0; i <= stacks; ++i) {
        float stackAngle = PI / 2 - i * stackStep;
        float xy = radius * cosf(stackAngle);
        float z = radius * sinf(stackAngle);

        for (int j = 0; j <= sectors; ++j) {
            float sectorAngle = j * sectorStep;
            float x = xy * cosf(sectorAngle);
            float y = xy * sinf(sectorAngle);
            vertices[vertexIndex++] = x;
            vertices[vertexIndex++] = y;
            vertices[vertexIndex++] = z;
        }
    }

    for (int i = 0; i < stacks; ++i) {
        int k1 = i * (sectors + 1);
        int k2 = k1 + sectors + 1;

        for (int j = 0; j < sectors; ++j, ++k1, ++k2) {
            if (i != 0) {
                indices[index++] = k1;
                indices[index++] = k2;
                indices[index++] = k1 + 1;
            }
            if (i != (stacks - 1)) {
                indices[index++] = k1 + 1;
                indices[index++] = k2;
                indices[index++] = k2 + 1;
            }
        }
    }
}

double lastFrame = 0.0;

double calculateFPS(GLFWwindow* window) {
    double currentFrame = glfwGetTime();
    double deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    return 1.0 / deltaTime;
}

Derivative evaluate(const State& initial, float dt, const Derivative& d) {
    State state;
    state.position = initial.position + d.dPosition * dt;
    state.velocity = initial.velocity + d.dVelocity * dt;

    Derivative output;
    output.dPosition = state.velocity;
    output.dVelocity = GRAVITY;
    return output;
}

void integrate(State& state, float dt) {
    Derivative a, b, c, d;

    a = evaluate(state, 0.0f, Derivative());
    b = evaluate(state, dt * 0.5f, a);
    c = evaluate(state, dt * 0.5f, b);
    d = evaluate(state, dt, c);

    glm::vec3 dPosition = (a.dPosition + 2.0f * (b.dPosition + c.dPosition) + d.dPosition) / 6.0f;
    glm::vec3 dVelocity = (a.dVelocity + 2.0f * (b.dVelocity + c.dVelocity) + d.dVelocity) / 6.0f;

    state.position = state.position + dPosition * dt;
    state.velocity = state.velocity + dVelocity * dt;
}

void handleWallCollision(State& sphere, const glm::vec3& boxMin, const glm::vec3& boxMax, float restitution) {
    for (int i = 0; i < 3; ++i) {
        if (sphere.position[i] - radius < boxMin[i]) {
            sphere.position[i] = boxMin[i] + radius;
            sphere.velocity[i] = -sphere.velocity[i] * restitution;
        }
        if (sphere.position[i] + radius > boxMax[i]) {
            sphere.position[i] = boxMax[i] - radius;
            sphere.velocity[i] = -sphere.velocity[i] * restitution;
        }
    }
}

void handleCollision(State& sphereA, State& sphereB, float restitution) {
    glm::vec3 delta = sphereB.position - sphereA.position;
    float distance = glm::length(delta);
    float minDistance = 2 * radius;

    if (distance < minDistance) {
        glm::vec3 normal = glm::normalize(delta);
        glm::vec3 relativeVelocity = sphereB.velocity - sphereA.velocity;

        float velocityAlongNormal = glm::dot(relativeVelocity, normal);
        if (velocityAlongNormal > 0)
            return;

        float j = -(1 + restitution) * velocityAlongNormal;
        j /= 1 / radius + 1 / radius;

        glm::vec3 impulse = j * normal;
        sphereA.velocity -= impulse / radius;
        sphereB.velocity += impulse / radius;
    }
}

int main() {
    const int numVertices = (sectors + 1) * (stacks + 1) * 3;
    const int numIndices = stacks * sectors * 6;
    GLfloat vertices[numVertices];
    GLuint indices[numIndices];
    generateVerticesAndIndices(vertices, indices);

    // Box vertices
    GLfloat boxVertices[] = {
        -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f
    };

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    double time = 0.0;

    GLFWwindow* window = glfwCreateWindow(width, height, "OpenGL", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    gladLoadGL();
    glViewport(0, 0, width, height);
    glEnable(GL_DEPTH_TEST);

    Shader shaderProgram("default.vert", "default.frag");
    GLuint uniID = glGetUniformLocation(shaderProgram.ID, "scale");

    VAO VAO1;
    VAO1.Bind();

    VBO VBO1(vertices, sizeof(vertices));
    EBO EBO1(indices, sizeof(indices));

    VAO1.LinkAttrib(VBO1, 0, 3, GL_FLOAT, 3 * sizeof(float), (void*)0);
    VAO1.Unbind();
    VBO1.Unbind();
    EBO1.Unbind();

    VAO boxVAO;
    boxVAO.Bind();
    VBO boxVBO(boxVertices, sizeof(boxVertices));
    boxVAO.LinkAttrib(boxVBO, 0, 3, GL_FLOAT, 3 * sizeof(GLfloat), (void*)0);
    boxVAO.Unbind();
    boxVBO.Unbind();

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    Camera camera(width, height, glm::vec3(0.0f, 0.0f, 2.0f));

    glm::vec3 spherePositions[spheresCount];
    std::vector<State> spheres(spheresCount);

    srand(time);
    for (int i = 0; i < spheresCount; ++i) {
        spheres[i].position = glm::vec3(
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f
        );

        spheres[i].velocity = glm::vec3(
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f
        );
    }

    float fov = 90.0f;
    float color[4] = { 0.8f, 0.7f, 0.3f, 1.0f };

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        shaderProgram.Activate();

        camera.Inputs(window);

        glm::vec3 boxMin(-1.0f, -1.0f, -1.0f);
        glm::vec3 boxMax(1.0f, 1.0f, 1.0f);

        // Apply RK4 integration
        for (int i = 0; i < spheresCount; ++i) {
            integrate(spheres[i], dt); // Update position using RK4
            handleWallCollision(spheres[i], boxMin, boxMax, restitution);
        
            for (int j = i + 1; j < spheresCount; ++j) {
                handleCollision(spheres[i], spheres[j], restitution);
            }
        }

        for (int i = 0; i < spheresCount; ++i) {
            glm::vec3 translation(spheres[i].position.x, spheres[i].position.y, spheres[i].position.z);
            camera.Matrix(fov, 0.1f, 100.0f, shaderProgram, "camMatrix", translation);
            VAO1.Bind();
            EBO1.Bind();
            glDrawElements(GL_TRIANGLE_STRIP, numIndices, GL_UNSIGNED_INT, 0);
        }

        float fps = calculateFPS(window);

        ImGui::Begin("My name is window, ImGUI window");
        ImGui::Text("FPS: ");
        ImGui::SameLine();
        ImGui::Text("%.0f", fps);

        if (ImGui::CollapsingHeader("Info")) {
            ImGui::Text("Window Dimensions:");
            ImGui::SameLine();
            ImGui::Text("%d x %d", width, height);

            std::string imgWindowPos = std::to_string(static_cast<int>(ImGui::GetWindowPos().x)) + "/" +
                std::to_string(static_cast<int>(ImGui::GetWindowPos().y));
            ImGui::Text("ImGui Window Position:");
            ImGui::SameLine();
            ImGui::Text("%s", imgWindowPos.c_str());
        }

        if (ImGui::CollapsingHeader("Timers")) {
            ImGui::Text("Frame Time:");
            ImGui::SameLine();
            ImGui::Text("%f", glfwGetTime() - lastFrame);
            ImGui::SameLine();
            ImGui::Text("ms");
        }

        if (ImGui::CollapsingHeader("Camera")) {
            ImGui::Text("Camera Position:");
            ImGui::SameLine();
            ImGui::Text("x: %f, y: %f, z: %f", camera.Position.x, camera.Position.y, camera.Position.z);

            ImGui::Text("Field of View");
            ImGui::SameLine();
            ImGui::SliderFloat("##FOV", &fov, 40, 120);
        }

        if (ImGui::CollapsingHeader("Spheres")) {
            ImGui::Text("Spheres:");
            ImGui::SameLine();
            ImGui::Text("%d", spheresCount);
            ImGui::ColorEdit4("Color", color);
            ImGui::SliderFloat("Restitution", &restitution, 0.0f, 1.0f); // Slider for restitution coefficient
        }

        ImGui::End();

        shaderProgram.Activate();
        glUniform1f(glGetUniformLocation(shaderProgram.ID, "size"), radius);
        glUniform4f(glGetUniformLocation(shaderProgram.ID, "color"), color[0], color[1], color[2], color[3]);

        // Draw the box
        boxVAO.Bind();
        camera.Matrix(fov, 0.1f, 100.0f, shaderProgram, "camMatrix", glm::vec3(0.0f));
        glDrawArrays(GL_LINES, 0, 24);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    VAO1.Delete();
    VBO1.Delete();
    EBO1.Delete();
    boxVAO.Delete();
    boxVBO.Delete();
    shaderProgram.Delete();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
