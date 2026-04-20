#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT float payload;
hitAttributeEXT vec2 bary;

void main() {
    vec3 bc = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    payload = dot(bc, vec3(0.299, 0.587, 0.114));
}
