//Made by Josue Galvan Ramos 
//reference https://www.shadertoy.com/view/lslXRS , https://www.shadertoy.com/view/ssBGRR

//PBR Shading Sampling

#define NEAR_CLIPPING_PLANE 0.1
#define FAR_CLIPPING_PLANE 40.0
#define NUMBER_OF_MARCH_STEPS 100
#define EPSILON 1e-4f
#define DISTANCE_BIAS 0.7
#define PI 3.14159265359
#define CAMERA_MOVING 1.0f

vec4 scene(vec3 position) {
    vec3 translate = vec3(0.0, 0.0, 11.0);
    vec3 sphere_pos = position - translate;
    float dist = length(sphere_pos) - 1.5;
   // vec4 color = vec4(dist, 0.98f, 0.756f, 0.403f); //gold color
   // vec4 color = vec4(dist, 0.44f, 0.44f, 0.44f); //grey color
    vec4 color = vec4(dist, 0.8f, 0.8f, 0.8f); //grey color
    return color;
}

vec3 normal(vec3 ray_hit_position, float smoothness) {	
// based on https://www.shadertoy.com/view/MdSGDW
    vec3 n;
    vec2 dn = vec2(smoothness, 0.0);
    n.x = scene(ray_hit_position + dn.xyy).x - scene(ray_hit_position - dn.xyy).x;
    n.y = scene(ray_hit_position + dn.yxy).x - scene(ray_hit_position - dn.yxy).x;
    n.z = scene(ray_hit_position + dn.yyx).x - scene(ray_hit_position - dn.yyx).x;
    return normalize(n);
}

vec4 raymarch(vec3 position, vec3 direction) {
    float total_distance = NEAR_CLIPPING_PLANE;
    vec4 result;
    for(int i = 0; i < NUMBER_OF_MARCH_STEPS; ++i) {
        result = scene(position + direction * total_distance);

        if(result.x < EPSILON)
            break;

        total_distance += result.x * DISTANCE_BIAS;

        // Stop if we are headed for infinity.
        // returns nothing on the color if we went to infinity.
        if(total_distance > FAR_CLIPPING_PLANE) {
            result.yzw = vec3(0.0, 0.0, 0.0);
            break;
        }
    }
    return vec4(total_distance, result.yzw);
}

////////////////////////
/// Shading
////////////////////////

///This is used for the specular contribution on the environment, we'll also use this later.
vec3 Specular_F_Roughness(vec3 specularColor, float a, vec3 h, vec3 v) {
    vec3 c = vec3(1.0f - a, 1.0f - a, 1.0f - a);
    return specularColor + (max(c, specularColor) - specularColor) * pow((1.0f - clamp(dot(v, h), 0.0f, 1.0f)), 5.0f);
}
float hash21(vec2 p,vec2 seed){  return fract(sin(dot(p,seed ))*43758.5453);}

float BrickGridMod(vec2 uv, out vec2 id,float lengt)
{
    vec2 pos = uv * vec2(1.0,lengt);
    pos.x += floor(uv.y*lengt)/lengt;
    id = floor(pos);
    id.y /= lengt;
    pos = fract(pos);
    vec2 uv2 = fract (pos)-0.5;
    uv2.y /= lengt;
    pos=abs(fract (pos + 0.5) - 0.5);
    float d = min(pos.x,pos.y/lengt);

    float y = length(uv2);

   return abs(d);
}
float DTermGGX(float a, float NdH) {
    //Isotropic ggx
    float a2 = a * a;
    float NdH2 = NdH * NdH;

    float denominator = NdH2 * (a2 - 1.0f) + 1.0f;
    denominator *= denominator;
    denominator *= PI;
    return a2 / denominator;
}

float GTermSmith(float a, float NdV, float NdL) {
	//smith schlick-GGX
    float k = a * 0.5f;
    float masking = NdV / (NdV * (1.0f - k) + k);
    float shadowing = NdL / (NdL * (1.0f - k) + k);
    return masking * shadowing;
}

vec3 FTermSchlick(vec3 specularColor, vec3 h, vec3 v) {
    return (specularColor + (1.0f - specularColor) * pow((1.0f - clamp(dot(v, h), 0.0f, 1.0f)), 5.0f));
}

vec3 CookTorrance(vec3 h, vec3 v, vec3 specularColor, float a, float NdL, float NdV, float NdH, vec3 F) {
    float D = DTermGGX(a, NdH);
    float G = GTermSmith(a, NdV, NdL);
    return (D * G * F) / (4.0 * NdL * NdV + EPSILON);
}

vec3 CookTorrance(vec3 h, vec3 v, vec3 specularColor, float a, float NdL, float NdV, float NdH, vec3 F,float D) {
    float G = GTermSmith(a, NdV, NdL);
    return (D * G * F) / (4.0 * NdL * NdV + EPSILON);
}
vec3 Lambert(vec3 albedo) {
    return albedo / PI;
}
vec3 SpecularLoveUWU(vec3 h, vec3 F, vec3 specularColor, vec3 normal, float roughness, vec3 lightPosition, vec3 lightDir, vec3 viewDir) {

    ///Calculate everything.
    float NdL = clamp(dot(normal, lightDir), 0.0f, 1.0f);
    float NdV = clamp(dot(normal, viewDir), 0.0f, 1.0f);

    float NdH = clamp(dot(normal, h), 0.0f, 1.0f);
    float VdH = clamp(dot(viewDir, h), 0.0f, 1.0f);
    float LdV = clamp(dot(lightDir, viewDir), 0.0f, 1.0f);
    float a = max(0.001f, roughness * roughness);
    return CookTorrance(h, viewDir, specularColor, a, NdL, NdV, NdH, F);
}
vec3 SpecularLoveUWU(vec3 h, vec3 F, vec3 specularColor, vec3 normal, float roughness, vec3 lightPosition, vec3 lightDir, vec3 viewDir,float D) {

    ///Calculate everything.
    float NdL = clamp(dot(normal, lightDir), 0.0f, 1.0f);
    float NdV = clamp(dot(normal, viewDir), 0.0f, 1.0f);

    float NdH = clamp(dot(normal, h), 0.0f, 1.0f);
    float a = max(0.001f, roughness * roughness);
    return CookTorrance(h, viewDir, specularColor, a, NdL, NdV, D, F);
}
float SheinLove(vec3 normal, vec3 viewDir, float sheenFactor, float sheenBias, float mask) {
    float NdV = clamp(dot(normal, viewDir), 0.0f, 1.0f);
    sheenFactor *= mask;
    return (1.0 - pow(NdV, sheenBias)) * sheenFactor;
}

vec3 thinFilm(float NdV) {
    vec3 a, b = vec3(0.5);
    vec3 c = vec3(1.0);
    vec3 d = vec3(0.0, 0.33, 0.66);

    return a + b + cos(2.0 + PI * (c * NdV + d));
}

float Danisotropic(vec3 Tanget, vec3 BiTanget, float roughness, vec3 h, vec3 normal) {
    float alphaT = 1.0 - roughness;
    float alphaB = 1.0 + roughness;
    float D1 = 1.0 / (PI * alphaT * alphaB + EPSILON);
    float Tdh2 = pow(clamp(dot(Tanget, h), 0.0f, 1.0), 2.0);
    float bdh2 = pow(clamp(dot(BiTanget, h), 0.0f, 1.0), 2.0);
    float ndh2 = pow(clamp(dot(normal, h), 0.0f, 1.0), 2.0);
    float D2 = 1.0 / (pow(Tdh2 / pow(alphaT, 2.0) + (bdh2 / pow(alphaB, 2.0)) + ndh2, 2.0));
    return D1 * D2;
}

vec3 hash(vec2 p, vec2 color) {
    vec3 q = vec3(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)), dot(p, color));
    return fract(sin(q) * 43758.5453);
}
mat2 makem2(in float theta){float c = cos(theta);float s = sin(theta);return mat2(c,-s,s,c);}
float noise( in vec2 x ){return texture(iChannel0, x*.01).x;}
vec2 gradn(vec2 p)
{
	float ep = .01;
	float gradx = noise(vec2(p.x+ep,p.y))-noise(vec2(p.x-ep,p.y));
	float grady = noise(vec2(p.x,p.y+ep))-noise(vec2(p.x,p.y-ep));
	return vec2(gradx,grady);
}

float flow(in vec2 p)
{
	float z=2.;
	float rz = 0.;
	vec2 bp = p;
	for (float i= 1.;i < 7.;i++ )
	{
		//primary flow speed
		p += iTime*0.01;
		
		//secondary flow speed (speed of the perceived flow)
		bp += iTime*0.01;
		
		//displacement field (try changing time multiplier)
		vec2 gr = gradn(i*p*.34+iTime*0.2);
		
		//rotation of the displacement field
		gr*=makem2(iTime*6.-(0.05*p.x+0.03*p.y)*0.1);
		
		//displace the system
		p += gr*.25;
		
		//add noise octave
		rz+= (sin(noise(p)*7.)*0.5+0.5)/z;
		
		//blend factor (blending displaced system with base system)
		//you could call this advection factor (.5 being low, .95 being high)
		p = mix(bp,p,.6);
		
		//intensity scaling
		z *= 1.2;
		//octave scaling
		p *= 1.5;
		bp *= 1.9;
	}
	return rz;	
}
vec3 iqnoise(in vec2 x) {
    vec2 p = floor(x);
    vec2 f = fract(x);

    float k = 1.0 + 60.0 * pow(0.98, 4.0);

    float va = 0.0;
    float wt = 0.0;

    float vaG = 0.0;
    float wtG = 0.0;

    //RED FORM
    for(int j = -2; j <= 2; j++) {
        for(int i = -2; i <= 2; i++) {
            vec2 g = vec2(float(i), float(j));
            vec3 o = hash(p + g, vec2(419.2, 371.9));
            vec2 r = g - f + o.xy;
            float d = dot(r, r);
            float ww = pow(1.0 - smoothstep(0.0, 1.8, sqrt(d)), k);
            va += o.z * ww;
            wt += ww;
        }
    }
   //GREEN FORMS
    for(int j = -2; j <= 2; j++) {
        for(int i = -2; i <= 2; i++) {
            vec2 g = vec2(float(i), float(j));
            vec3 o = hash(p + g, vec2(371.9, 419.2));
            vec2 r = g - f + o.xy;
            float d = dot(r, r);
            float ww = pow(1.0 - smoothstep(0.0, 1.8, sqrt(d)), k);
            vaG += o.z * ww;
            wtG += ww;
        }
    }

    float RED = va / wt;
    float GREEN = vaG / wtG;
    return vec3(RED, GREEN, 1);
}

vec3 proceduralGlintNormal(vec2 uv, float scale) {
    uv.x *= iResolution.x / iResolution.y;
    uv *= scale;
    return iqnoise(uv);
}
///NORMAL
vec3 ComputeLight(vec3 albedoColor, vec3 specularColor, vec3 normal, float roughness, vec3 lightPosition, vec3 lightColor,
 vec3 lightDir, vec3 viewDir, float met,bool thin,vec3 h) {

    vec3 tangent = vec3(normal.y,-normal.x,0);
    vec3 bitangent = cross(normal,tangent);

    h *= normalize(lightDir + viewDir);
    float NdV = clamp(dot(normal, viewDir), 0.0f, 1.0f);

    vec3 F = FTermSchlick(specularColor, h, viewDir);

    vec3 Thin = thinFilm(NdV);
    if(thin)F *= Thin;
    ///Get the diffuse result and the specular result.
    vec3 ColorDiffuse = Lambert(albedoColor);
    vec3 ColorSpecular = SpecularLoveUWU(h, F, specularColor, normal, roughness, lightPosition, 
    lightDir, viewDir); 

    ///Diffuse and Specular are mutually exclusive, if light goes into diffuse it is because it was not reflected 
    vec3 diffuseContribution = vec3(1.0f, 1.0f, 1.0f) - F; ///To get our diffuse contribution we substract the specular contribution from a white color.

    vec3 SpecDif = vec3(1.0) - FTermSchlick(vec3(0.4), h, viewDir);
    
    ColorDiffuse *= diffuseContribution;
    ColorDiffuse *= 1.0f - met;

    return lightColor * (ColorDiffuse + ColorSpecular) * SpecDif ;
}

//SHEEN OVERRIDE
vec3 ComputeLight(vec3 albedoColor, vec3 specularColor, vec3 normal, float roughness, vec3 lightPosition, vec3 lightColor,
vec3 lightDir, vec3 viewDir, float met, float sheen, float sheenBias, float mask) {

    vec3 h = normalize(lightDir + viewDir);

    vec3 F = FTermSchlick(specularColor, h, viewDir);
    ///Get the diffuse result and the specular result.
    vec3 ColorDiffuse = Lambert(albedoColor);

    vec3 ColorSpecular = SpecularLoveUWU(h, F, specularColor, normal, roughness, lightPosition, lightDir, viewDir); 

    ///Diffuse and Specular are mutually exclusive, if light goes into diffuse it is because it was not reflected 
    vec3 diffuseContribution = vec3(1.0f, 1.0f, 1.0f) - FTermSchlick(specularColor, h, viewDir); ///To get our diffuse contribution we substract the specular contribution from a white color.

    vec3 SpecDif = vec3(1.0) - FTermSchlick(vec3(0.4), h, viewDir);
    ColorDiffuse *= diffuseContribution;
    ColorDiffuse *= 1.0f - met;

    float Sheenvar = SheinLove(normal, viewDir, sheen, sheenBias, mask);
    ///Now we just multiply the NdL by the lightcolor and by the colorDiffuse and ColorSpecular
    return lightColor * (ColorDiffuse + ColorSpecular) * SpecDif + Sheenvar;
}

//COAT OVERRIDE
vec3 ComputeLight(vec3 albedoColor, vec3 specularColor, vec3 normal, float roughness, vec3 lightPosition, vec3 lightColor,
vec3 lightDir, vec3 viewDir, float met, vec3 coatNormal, float coatRoughness, vec3 h) {
    h *= normalize(lightDir + viewDir);

    float NdV = clamp(dot(normal, viewDir), 0.0f, 1.0f);
    float NdVCoat = clamp(dot(coatNormal, viewDir), 0.0f, 1.0f);

    vec3 coatColor = vec3(0.4);

    vec3 F = FTermSchlick(specularColor, h, viewDir);
    vec3 coatF = FTermSchlick(coatColor, h, viewDir);
    vec3 Thin = thinFilm(NdV);
    vec3 ThinCoat = thinFilm(NdVCoat);

    F *= Thin;
    coatF *= ThinCoat;

    ///Get the diffuse result and the specular result.
    vec3 ColorDiffuse = Lambert(albedoColor);
    vec3 ColorSpecular = SpecularLoveUWU(h, F, specularColor, normal, roughness, lightPosition, lightDir, viewDir);
    vec3 ClearCoat = SpecularLoveUWU(h, coatF, coatColor, coatNormal, coatRoughness, lightPosition, lightColor, lightDir);

    ///Diffuse and Specular are mutually exclusive, if light goes into diffuse it is because it was not reflected 
    vec3 diffuseContribution = vec3(1.0f, 1.0f, 1.0f) - F; ///To get our diffuse contribution we substract the specular contribution from a white color.

    vec3 SpecDif = vec3(1.0) - FTermSchlick(vec3(0.4), h, viewDir);
    ColorDiffuse *= diffuseContribution;
    ColorDiffuse *= 1.0f - met;
    ///Now we just multiply the NdL by the lightcolor and by the colorDiffuse and ColorSpecular

    return lightColor * (ColorDiffuse + ColorSpecular) * (SpecDif + ClearCoat);
}
//OVERRIDE DANISOTROPIC
vec3 ComputeLight(vec3 albedoColor, vec3 specularColor, vec3 normal, float roughness, vec3 lightPosition, vec3 lightColor,
 vec3 lightDir, vec3 viewDir, float met,bool thin,vec3 tangent,vec3 bitangent) {
    
    vec3 h = normalize(lightDir + viewDir);
    float NdV = clamp(dot(normal, viewDir), 0.0f, 1.0f);

    vec3 F = FTermSchlick(specularColor, h, viewDir);

    vec3 Thin = thinFilm(NdV);
    if(thin)F *= Thin;
    ///Get the diffuse result and the specular result.
    vec3 ColorDiffuse = Lambert(albedoColor);

    float D = Danisotropic(tangent,bitangent,roughness,h,normal);
    vec3 ColorSpecular = SpecularLoveUWU(h, F, specularColor, normal, roughness, lightPosition, 
    lightDir, viewDir,D); 

    ///Diffuse and Specular are mutually exclusive, if light goes into diffuse it is because it was not reflected 
    vec3 diffuseContribution = vec3(1.0f, 1.0f, 1.0f) - F; ///To get our diffuse contribution we substract the specular contribution from a white color.

    vec3 SpecDif = vec3(1.0) - FTermSchlick(vec3(0.4), h, viewDir);
    ColorDiffuse *= diffuseContribution;
    ColorDiffuse *= 1.0f - met;

    ///Now we just multiply the NdL by the lightcolor and by the colorDiffuse and ColorSpecular

    return lightColor * (ColorDiffuse + ColorSpecular) * SpecDif;
}
vec3 ToLinear(vec3 c) {
    c.x = pow(c.x, 2.2f);
    c.y = pow(c.y, 2.2f);
    c.z = pow(c.z, 2.2f);

    return c;
}

vec3 TosRGB(vec3 c) {
    c.x = pow(c.x, 1.0f / 2.2f);
    c.y = pow(c.y, 1.0f / 2.2f);
    c.z = pow(c.z, 1.0f / 2.2f);

    return c;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ///Prepare everything for raymarching.

    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord / iResolution.xy;

    // Get the positions for the ray.
    vec2 pos = uv * 2.0f - 1.0f;
    pos.x *= iResolution.x / iResolution.y;

    ///Build the direction of the ray.
    vec3 direction = normalize(vec3(pos, 2.5));

    ///The camara origin.
    vec3 camera_origin = vec3(0.0, 0.0, 4.0); 

    ///Do the raymarch.
    vec4 result = raymarch(camera_origin, direction); 

    ///get the intersection or the world position of the pixel.
    vec3 pixelWorldPos = camera_origin + direction * result.x;

    ///Use that to get the normal of the surface.
    vec3 n = normal(pixelWorldPos, 0.01);

    ///Get the viewDir which is the direction from the camera to the pixel world position.
    ///This is the same as the direction, but here's how to calculate it anyway.
    vec3 viewDir = normalize(pixelWorldPos - camera_origin);

    ///This variable is used so the background does not react to roughness.
    float envRoughness = 1.0f;

    ///If we didn't hit anything.
    if(result.x > FAR_CLIPPING_PLANE) {
        ///Set the normal to nothing.
        n = vec3(0.0f, 0.0f, 0.0f);
        ///Make it so that we don't sample the background by roughness.
        envRoughness = 0.0f;
    }

    /////////////////////////////////////////
    ///EXPERIMENTS / CHANGEABLE VALUES 
    /////////////////////////////////////////

    ///Set the light Dir
    vec3 lightDir = vec3(0.0f, -4.0f, 5.0f); //MADE MODIFICATIONS TO LIGHT
    ///Every direction is always normalized, always.
    lightDir = normalize(lightDir);

    ///The light color.
    float lightIntensity = 2.0;
    vec3 lightColor = vec3(0.77f, 0.67f, 0.54f);
    lightColor *= lightIntensity;

    ///The roughness. It is currently changing according to a sin. you can change it here.
    float roughness = clamp((sin(iTime) + 1.0f) / 2.0f, 0.0f, 1.0f);

    /////////////////////////////////////////
    /// Calculations
    /////////////////////////////////////////

    ///Now we get the color from the hit surface. This is the material color.
    ///Known as Albedo.
    vec3 col = result.yzw;

    //new values of u & v 
    float u = atan(n.z, n.x) / PI * 2.0 + iTime / 8.0;
    float v = asin(n.y) / PI * 2.0 + 0.5;

    ///We need to transform the color from sRGB (display values) to linear (Math correct pro values).
    ///https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-24-importance-being-linear

    col = ToLinear(col); // original color 

    
///////////////////////////////////// COMENT THIS ONES WHEN UNCOMMENT OTHERS
    float met = 1.0;
    //roughness = 0.3;
///Get the specular color by lerping from a blackish color to the albedo color using the metallic as alpha.
///If something is 100% metallic, then its specular color will be the full albedo. 
///If it is 0% metallic, there will barely be any specular color.
    vec3 specColor = mix(vec3(0.06f, 0.06f, 0.06f), col, met);
    vec3 h;
    vec3 pbr = ComputeLight(col, specColor, n, roughness, -lightDir * 1000.0f, lightColor, -lightDir, -viewDir, met,false,h);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////Example1
/*
    vec3 alb = texture(iChannel5,vec2(u,v)).xyz;
    vec3 ao = texture(iChannel6,vec2(u,v)).xyz;

    //new declaration of met using the texture mask
    met = texture(iChannel7,vec2(u,v)).x; 

    //declaration of the norm texture
    vec3 norm = texture(iChannel8,vec2(u,v)).xyz;

    //roughness using the texture as mask | comment to see the animation #Animation does not follow texture.. yet
    roughness = texture(iChannel9,vec2(u,v)).x;

    // calculation of the normal 
    vec3 tangent = vec3(n.y,-n.x,0);
    vec3 bitangent = cross(n,tangent);
    norm = normalize(norm*2.0-1.0);
    mat3 matrix  = mat3(tangent,bitangent,norm);
    n*= normalize(matrix*norm);

    //adding the albedo to the color
    col*=alb;
    //adding ambien occlusion to albedo
    col*=ao;

    //TEXTURE CHANNELS NORMALS 

    #iChannel5 "file://SpaceCruiser/space-cruiser-panels2_albedo.png"
    #iChannel6 "file://SpaceCruiser/space-cruiser-panels2_ao.png"
    #iChannel7 "file://SpaceCruiser/space-cruiser-panels2_metallic.png"
    #iChannel8 "file://SpaceCruiser/space-cruiser-panels2_normal-ogl.png"
    #iChannel9 "file://SpaceCruiser/space-cruiser-panels2_roughness.png"

    specColor = mix(vec3(0.06f, 0.06f, 0.06f), col, met);
    pbr = ComputeLight(col, specColor, n, roughness, -lightDir * 1000.0f,
    lightColor, -lightDir, -viewDir, met,true);
*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////Example 2
/*
    vec3 alb = texture(iChannel6,vec2(u,v)).xyz;
    vec3 norm = texture(iChannel8,vec2(u,v)).xyz;

    // calculation of the normal 
    vec3 tangent = vec3(n.y,-n.x,0);
    vec3 bitangent = cross(n,tangent);
    norm = normalize(norm*2.0-1.0);
    mat3 matrix  = mat3(tangent,bitangent,norm);
    n*= normalize(matrix*norm);
    //adding the albedo to the color
    col*=alb;

    //mask to shein
    float txt = texture(iChannel5,vec2(u,v)).x; 
        
    /////////////////////////////////////////////////////////  TEXTURES

    #iChannel5 "file://Suede/Suede_Sheen2.jpg"
    #iChannel6 "file://Suede/Suede_Diffuse.jpg"
    #iChannel7 "file://Suede/Suede_normals.jpg"
    #iChannel8 "file://Suede/Suede_normals_02.jpg"
    #iChannel9 "file://Suede/Suede_Sheen.jpg"

    specColor = mix(vec3(0.06f, 0.06f, 0.06f), col, met);
 pbr = ComputeLight(col, specColor, n, roughness, -lightDir *
 1000.0f, lightColor, -lightDir, -viewDir, met,2.0,0.5,txt); 
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////Example 3
 /*
    vec3 coatN = n;

    vec3 alb = texture(iChannel5,vec2(u,v)).xyz;
    vec3 hei = texture(iChannel5,vec2(u,v)).xyz; 
    vec3 norm = texture(iChannel8,vec2(u,v)).xyz;
    met *=  texture(iChannel7,vec2(u,v)).x;
    roughness *= texture(iChannel9,vec2(u,v)).x;

    // calculation of the normal 
    vec3 tangent = vec3(n.y,-n.x,0);
    vec3 bitangent = cross(n,tangent);
    norm = normalize(norm*2.0-1.0);
    mat3 matrix  = mat3(tangent,bitangent,norm);
    n*= normalize(matrix*norm);

    //adding the albedo to the color
    col*=alb;

    /////////////////////////////////////////////////////////// TEXTURES  
    
    #iChannel5 "file://woodparquet_84-1K/woodparquet_84_basecolor-1K.png"
    #iChannel6 "file://woodparquet_84-1K/woodparquet_84_height-1K.png"
    #iChannel7 "file://woodparquet_84-1K/woodparquet_84_metallic-1K.png"
    #iChannel8 "file://woodparquet_84-1K/woodparquet_84_normal-1K.png"
    #iChannel9 "file://woodparquet_84-1K/woodparquet_84_roughness-1K.png"
    #iChannel10 "file://woodparquet_84-1K/woodparquet_84_ambientocclusion-1K.png"

    specColor = mix(vec3(0.06f, 0.06f, 0.06f), col, met);    
    pbr = ComputeLight(col, specColor, n, roughness, -lightDir * 1000.0f, 
    lightColor, -lightDir, -viewDir, met,coatN,1.0,hei); 
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////Example 4
/*
   // roughness = 0.4;
    vec3 norm = proceduralGlintNormal(vec2(u,v),10.0);

    // calculation of the normal 
    vec3 tangent = vec3(n.y,-n.x,0);
    vec3 bitangent = cross(n,tangent);
    norm = normalize(norm*2.0-1.0);
    mat3 matrix  = mat3(tangent,bitangent,norm);
    n*= normalize(matrix*norm);

    specColor = mix(vec3(0.06f, 0.06f, 0.06f), col, met); 
    pbr = ComputeLight(col, specColor, n, roughness,
    -lightDir * 1000.0f, lightColor, -lightDir, -viewDir,
     met,true,tangent,bitangent);
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////Example 5
/*
    vec3 norm = vec3(0,0,0);
    // calculation of the normal 
    vec3 tangent = vec3(n.y,-n.x,0);
    vec3 bitangent = cross(n,tangent);
    norm = normalize(norm*2.0-1.0);
    mat3 matrix  = mat3(tangent,bitangent,norm);
    n*= normalize(matrix*norm);

    specColor = mix(vec3(0.06f, 0.06f, 0.06f), col, met); 
    pbr = ComputeLight(col, specColor, n, roughness, -lightDir * 1000.0f, lightColor, -lightDir, -viewDir, met,false,tangent,bitangent);
*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// FINAL

    vec2 uvs = vec2(u,v);
    uvs.x *= iResolution.x/iResolution.y;
    float scale = 2.0;
    uvs*=scale;
    
    #iChannel0 "file://Suede/Suede_normals.jpg"
    #iChannel6 "file://Suede/Suede_Diffuse.jpg"
    #iChannel7 "file://Suede/Suede_normals.jpg"
    //Used to save vec out of function
    vec2 id;

    //ALBEDO COLORS
    vec3 shapecolor1  = vec3(0.18, 0.04, 0.04);
	vec3 shapecolor2  = vec3(0.41, 0.14, 0.06);

    // LINE SIZE
    float LineSize = 0.065;
    float LineSize2 = 0.01; //IF LINESIZE2 > LINESIZE IS GOING TO IVERT COLORS


    //2nd COLOR VARIATION
    float speed=0.4;
    float AreaIntensity = 5.0;


    //BRICKSHAPES
    float absD = BrickGridMod(uvs,id,2.0);
    float rz = flow(vec2(id));

    //LINES COLOR
    vec3 colorLine = vec3(0.25, 0.06, 0.03);
    vec3 Lcol =colorLine/rz;
    Lcol=pow(Lcol,vec3(1.5));
    //HEIGH MAP
    vec3 hCol = vec3(0.0);
    vec3 hCol2 = vec3(1.0);

    //BLEND FOR HASH AND ANIMATION COLOR
    float blend = pow(abs(sin(hash21(id,vec2(419.2,371.9))*4.35 + iTime*speed)),AreaIntensity);	
    //USED FOR INVERTED HASH TO CREATE NORMALMAP, NO ANIMATED MAPS
    float nblend = pow(abs(sin(hash21(id,vec2(371.9,419.2))*4.35)), AreaIntensity);	
    float nblend2 = pow(abs(sin(hash21(id,vec2(419.2,371.9))*4.35)),AreaIntensity);
    //NORMAL MAP
    vec3 normalMap = vec3(nblend,nblend2,1);
    //ALBEDO COLOR
    shapecolor1 = mix(shapecolor1,shapecolor2,blend); 
    //HEIGH MAP
    vec3 hcolor = mix(hCol,hCol2,nblend);

    
    //FINAL COLORS
    col *= mix(Lcol,shapecolor1,smoothstep(LineSize2, LineSize, absD)); 
    vec3 norm = mix(vec3(absD),normalMap,smoothstep(LineSize2, LineSize,absD));
    h *= mix(vec3(absD),hcolor,smoothstep(LineSize2, LineSize, absD)); 

    //Using textures
    col*=texture(iChannel6,vec2(u,v)).xyz;
    roughness = texture(iChannel7,vec2(u,v)).x+0.1;
    met = 0.1;
   
    // calculation of the normal norm 
    vec3 tangent = vec3(n.y,-n.x,0);
    vec3 bitangent = cross(n,tangent);
    norm = normalize(norm*2.0-1.0);
    mat3 matrix  = mat3(tangent,bitangent,norm);
    n*= normalize(matrix*norm);

    
    fragColor = vec4(col,1.0);

    specColor = mix(vec3(absD), col, met); 

    pbr = ComputeLight(col, specColor, n, roughness, -lightDir * 1000.0f,
    lightColor, -lightDir, -viewDir, met,false,h);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///Compute the actual DIRECT light contribution.
    ///Direct lighting is the contribution of real lights directly hitting a surface. 
    ///Indirect lighting is light that comes from other surfaces and the light that bounces off of them.
    ///We send the lightdir negated and the view dir negated because they have to follow the same direction as the normal.
    ///So the dot products work.

    

    

    ///This is IBL or Image based Lighting. Which means we use a texture or cubemap's data as light contribution
    ///from the environment. 

    ///Now we calculate the fresnel for the environment.
    vec3 envFresnel = Specular_F_Roughness(specColor.xyz, roughness * roughness, n, -viewDir).xyz;

    ///We calculate again the specular and diffuse contributions because there are two parts 
    ///There's two parts the environment and the irradiance. 
    ///One affects the diffuse and the other one affects the specular.
    vec3 Kd = 1.0f - envFresnel;
    Kd *= 1.0f - met;

    ///Get the reflection vector using the view dir on the normal.
    vec3 reflection = reflect(viewDir, n);

    ///We are gonna sample the cubemap with a sample level depending of the roughness of the surface.
    ///The mip maps help us simulate the environment reflections becoming diffused depending of thr surface's roughness.
    ///We have 9.0f mipmaps so we multiply roughness which goes from 0.0f to 1.0f by 9.
    ///This means that when roughness is 0.0f, we'll get the "clearest" reflection.
    ///And when it is 1.0f, we'll get a diffuse reflection.
    float sampleLevel = roughness * 9.0f * envRoughness;

    /////////////////////////////////////////////////////////// CHANELS
    #iChannel1 "file://cubemap/Church_{}.jpg"
    #iChannel1::Type "CubeMap"
    #iChannel1::MinFilter "LinearMipMapLinear"

    #iChannel2 "file://cubemap/Forest_{}.png"
    #iChannel2::Type "CubeMap"
    #iChannel2::MinFilter "LinearMipMapLinear"

    #iChannel3 "file://cubemap/Plaza_{}.jpg"
    #iChannel3::Type "CubeMap"
    #iChannel3::MinFilter "LinearMipMapLinear"

    #iChannel4 "file://cubemap/Hall_{}.png"
    #iChannel4::Type "CubeMap"
    #iChannel4::MinFilter "LinearMipMapLinear"


    

    vec4 env = textureLod(iChannel1, reflection, sampleLevel);

    //vec4 env = vec4(1.0,1.0,1.0,1.0);
    ///Make it linear.
    env.xyz = ToLinear(env.xyz);

    ///Now we will sample the same cubemap, but with the largest mip. 
    ///We use the normal this time, because we want to get a value for irradiance.
    ///This is just light that "bleeds" into other things which are nearby.
    ///This is not a reflection. Think about it like this. You're standing next to a red wall.
    ///The wall is hit by the sunlight and your body looks red from standing near to it.
    ///It is not a reflection, it does not change if you move or look at your skin from a different angle.
    ///It's just the wall irradiating red light and coloring your skin.
    vec4 irr = textureLod(iChannel1, n, 9.0f);

    irr.xyz = ToLinear(irr.xyz);

    ///Finally, let's output to screen.
    if(result.x > FAR_CLIPPING_PLANE) {
        ///IF we hit nothing, just print the environment color.
        fragColor = vec4(env.xyz, 1.0f);
    } else {
        ///If we did hit something. Do the final calculation.
        vec3 directLight = pbr;

        ///Multiply the surface color by the irradiance and by the diffuse contribution.
        vec3 ambientLight = col * irr.xyz * Kd; 

        ///Multiply the environment by the fresnel.
        ///The environment should be more reflective on grazing angles.
        vec3 specularIndirectLight = env.xyz * envFresnel;

        fragColor = vec4(directLight + ambientLight + specularIndirectLight, 1.0f);
    }

    ///We are done with calculations so just go back to sRGB for our screens to display.
    fragColor.xyz = TosRGB(fragColor.xyz);
}