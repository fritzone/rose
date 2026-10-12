// rose.cpp — The code for the Linux Magazine Article
//
// Build:  g++ -O2 rose.cpp -o rose3d $(sdl2-config --cflags --libs)

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdlib>

#ifndef HEADLESS
#include <SDL2/SDL.h>
#endif

// ----------------------------------------------------------------------------
// small 3D math
// ----------------------------------------------------------------------------
static const double PI = 3.14159265358979323846;

struct V3 {
    double x=0,y=0,z=0;
    V3(){} V3(double a,double b,double c):x(a),y(b),z(c){}
};
static inline V3 operator+(V3 a,V3 b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
static inline V3 operator-(V3 a,V3 b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
static inline V3 operator*(V3 a,double s){ return {a.x*s,a.y*s,a.z*s}; }
static inline double dot(V3 a,V3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline V3 cross(V3 a,V3 b){ return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
static inline double len(V3 a){ return std::sqrt(dot(a,a)); }
static inline V3 norm(V3 a){ double l=len(a); return l>1e-12? a*(1.0/l) : V3(0,0,1); }
static inline V3 mix(V3 a,V3 b,double t){ return a+(b-a)*t; }
static inline double clampd(double x,double lo,double hi){ return x<lo?lo:(x>hi?hi:x); }
static inline double smoothstep(double a,double b,double x){
    if(a==b) return x<a?0.0:1.0;
    double t=clampd((x-a)/(b-a),0.0,1.0);
    return t*t*(3.0-2.0*t);
}
static inline V3 mul(V3 a,V3 b){ return {a.x*b.x,a.y*b.y,a.z*b.z}; }

// nudge a colour in HSV space: hue by dh degrees, saturation and value scaled
static V3 hsvAdjust(V3 c, double dh, double sMul, double vMul){
    if(dh==0.0 && sMul==1.0 && vMul==1.0) return c;
    double mx=std::max({c.x,c.y,c.z}), mn=std::min({c.x,c.y,c.z}), d=mx-mn;
    double h=0, sat= mx>1e-9 ? d/mx : 0, val=mx;
    if(d>1e-9){
        if(mx==c.x)      h=60.0*std::fmod((c.y-c.z)/d, 6.0);
        else if(mx==c.y) h=60.0*((c.z-c.x)/d + 2.0);
        else             h=60.0*((c.x-c.y)/d + 4.0);
    }
    h=std::fmod(h+dh+720.0, 360.0);
    sat=clampd(sat*sMul,0,1); val=clampd(val*vMul,0,1);
    double C=val*sat, X=C*(1.0-std::fabs(std::fmod(h/60.0,2.0)-1.0)), m=val-C;
    V3 r = h<60 ? V3(C,X,0) : h<120 ? V3(X,C,0) : h<180 ? V3(0,C,X)
         : h<240 ? V3(0,X,C) : h<300 ? V3(X,0,C) : V3(C,0,X);
    return r + V3(m,m,m);
}

static inline uint64_t hash64(uint64_t x){ // splitmix64 finaliser
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x>>30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x>>27)) * 0x94D049BB133111EBull;
    return x ^ (x>>31);
}
static inline double hashUnit(uint64_t h,int k){ return (double)((h>>(k*16))&0xFFFF)/65536.0; }

struct Vert { V3 p, n, col; double u=0, v=0;      // position, normal, colour, petal UV
             double ga=0, gb=0;                   // mesh-grid coords in [0,1] (for the wireframe)
             double ao=1; };                      // ambient occlusion: 1 = open sky, 0 = buried
struct Tri  { Vert a,b,c; int mat=0;              // mat: 0 = petal, 1 = green plant part
             double na=1, nb=1;                   // grid cells along ga / gb of the source mesh
             int mesh=0; };                       // which mesh (rose, stem, a leaf...) it belongs to
static std::vector<Tri> g_tris;            // the whole scene as a triangle soup

// give every triangle added since `start` a fresh mesh id
static int g_nextMeshId=0;
static void tagNewMesh(size_t start){
    ++g_nextMeshId;
    for(size_t i=start;i<g_tris.size();++i) g_tris[i].mesh=g_nextMeshId;
}

static double          RANDOMNESS = 1.0;
static uint64_t        g_seed = 1;
static std::mt19937_64 g_rng(1);
static double rnd(double a,double b){ return std::uniform_real_distribution<double>(a,b)(g_rng); }
static double vary(double x,double frac){ return x*(1.0 + RANDOMNESS*rnd(-frac,frac)); }  // x ± frac

static V3     g_bloomMin, g_bloomMax;      // bounding box of the bloom (for auto-framing)
static V3     g_center;                    // camera look-at point
static V3     g_focusPt;                   // centre of the bloom (depth of field focuses here)
static double g_bloomR=1.0, g_bloomTopY=1.0;
static double g_camDist = 4.6;             // camera distance (auto-computed)
static const double FOV = 42.0*PI/180.0;   // vertical field of view

struct RoseParams {
    int    numPetals = 48;
    double petalFreq = 4.2;
    double condense  = 4.8;
    double droop     = 1.95;
    double notch     = 0.5;
    double rippleWaves = 3.0;
    double twist     = 0.0;
    double coreStart = 0.20;
    double coreFull  = 0.70;
    double coreNotch = 0.12;
    double coreOpen  = 0.25;
    double coreSize  = 0.80;
    int    resAround = 26;
    int    resAcross = 22;
};
static RoseParams RP;                      // working copy: the defaults above + random variation

// per-petal random variation, interpolated smoothly along the spiral so that
// neighbouring petals blend into each other without cracks
struct PetalVar { double size=1, droop=1, open=0, ripple=0, shade=1, hue=0, curl=0, twist=0; };
static std::vector<PetalVar> g_petalVar;   // one per petal, plus one on either end

static PetalVar petalVarAt(double th){
    const int n=(int)g_petalVar.size();
    if(n<2) return PetalVar{};
    double q=RP.petalFreq*th/(2.0*PI) - 0.5;          // petal centres fall on integers
    double k=std::floor(q), f=q-k; f=f*f*(3.0-2.0*f);
    const PetalVar& a=g_petalVar[std::clamp((int)k+1,0,n-1)];
    const PetalVar& b=g_petalVar[std::clamp((int)k+2,0,n-1)];
    auto m=[f](double x,double y){ return x+(y-x)*f; };
    return { m(a.size,b.size), m(a.droop,b.droop), m(a.open,b.open), m(a.ripple,b.ripple), m(a.shade,b.shade),
             m(a.hue,b.hue), m(a.curl,b.curl), m(a.twist,b.twist) };
}

static void randomizeBloom(){
    RP = RoseParams{};
    RP.numPetals = std::max(12, RP.numPetals + (int)std::lround(RANDOMNESS*rnd(-6.0,6.0)));
    RP.petalFreq = vary(RP.petalFreq, 0.04);
    RP.condense  = vary(RP.condense,  0.07);
    RP.droop     = vary(RP.droop,     0.12);
    RP.notch     = vary(RP.notch,     0.15);
    RP.rippleWaves = RP.rippleWaves + RANDOMNESS*rnd(-1.0,1.5);
    RP.twist     = RP.twist + RANDOMNESS*rnd(-0.10,0.10);
    RP.coreNotch = vary(RP.coreNotch, 0.30);
    RP.coreOpen  = vary(RP.coreOpen,  0.20);
    RP.coreSize  = vary(RP.coreSize,  0.06);
    const int n = RP.numPetals+2;
    g_petalVar.assign(n, PetalVar{});
    for(int k=0;k<n;++k){
        PetalVar& v = g_petalVar[k];
        double outer = 1.0 - (double)k/(n-1);          // 1 = outermost guard petal
        v.size   = vary(1.0, 0.06);
        v.droop  = vary(1.0, 0.18);
        v.open   = RANDOMNESS*rnd(-0.05, 0.05);       // radians of extra tilt
        v.ripple = RANDOMNESS*rnd( 0.00, 0.04);
        v.shade  = vary(1.0, 0.07);
        v.hue    = RANDOMNESS*rnd(-3.0, 3.0);         // degrees
        v.curl   = RANDOMNESS*rnd( 0.00, 0.14)*outer*outer;   // outer petals roll their rims back
        v.twist  = RP.twist + RANDOMNESS*rnd(-0.05, 0.05);
    }
}

// ----------------------------------------------------------------------------
// Nylander parametric rose surface
//   parameters: th (winds around, making the spiral of petals), x in [0,1]
// ----------------------------------------------------------------------------
static V3 rosePoint(double th, double x){
    const PetalVar pv = petalVarAt(th);
    const double depth = th / (RP.numPetals*2.0*PI/RP.petalFreq);      // 0 outer -> 1 centre
    const double core  = smoothstep(RP.coreStart, RP.coreFull, depth);  // 0 .. 1 in the heart
    double phi   = (PI/2.0) * std::exp(-th / (RP.condense*PI));         // petals tilt up as we spiral in
    const double floorPhi = RP.coreOpen*std::max(0.0, 1.15-depth);      // soft, still-shrinking floor
    phi = std::sqrt(phi*phi + floorPhi*floorPhi);
    phi = clampd(phi + pv.open, 0.01, PI/2.0 + 0.2);
    const double notch = RP.notch*(1.0 + (RP.coreNotch-1.0)*core);     // rims merge into a scroll
    double petal = 1.0 - notch * std::pow( (5.0/4.0) *
                   std::pow(1.0 - std::fmod(RP.petalFreq*th, 2.0*PI)/PI, 2.0) - 0.25, 2.0); // ruffled edge
    double hang  = pv.droop * RP.droop * std::pow(x,2.0) * std::pow(x-1.0,2.0) * std::sin(phi); // petal droop
    double rollB = smoothstep(0.55,1.0,x);
    hang += pv.curl * rollB*rollB * std::sin(phi);                                   // rim rolled back
    double r     = petal * ( x * std::sin(phi) + hang * std::cos(phi) );
    r *= 1.0 + pv.ripple * x*x*x * std::sin(2.0*PI*RP.rippleWaves*RP.petalFreq*th/(2.0*PI)); // wavy rim
    double ang = th + pv.twist*x*x;                                                  // sideways sweep
    double X = r * std::sin(ang);
    double Y = r * std::cos(ang);
    double Z = petal * ( x * std::cos(phi) - hang * std::sin(phi) );
    // map Nylander axis (Z) to world up (Y) so the bloom opens upward
    return V3(X, Z, Y)*(pv.size*(1.0 + (RP.coreSize-1.0)*core));
}

struct ColorStop { double pos; V3 rgb; };

struct PetalPalette {
    std::vector<ColorStop> stops = {     // default: deep crimson throat -> bright rose rim
        {0.0, V3{0.42, 0.02, 0.09}},
        {1.0, V3{0.86, 0.10, 0.20}},
    };
    double axisRadial = 1.0;     // blend along base->edge of each petal
    double axisDepth  = 0.0;     // blend along outer->inner petals of the bloom
    double bias       = 0.7;     // <1 pushes colour toward the rim, >1 toward the throat
    // picotee — distinct colour at the very edge of each petal
    bool   edgeEnable = false;
    V3     edgeColor  = V3{0.85, 0.05, 0.12};
    double edgeWidth  = 0.22;    // fraction of the petal (from rim inward) taking the edge colour
    // bicolour reverse — different colour on the back of petals
    bool   reverseEnable = false;
    V3     reverseColor  = V3{0.95, 0.85, 0.55};
    // throat glow — many roses are yellow or cream at the very base of the petals
    double throatAmt   = 0.0;
    V3     throatColor = V3{0.95, 0.80, 0.30};
    // ageing — the tips of the outermost petals brown a little as the bloom opens
    double ageAmt      = 0.0;
    V3     ageColor    = V3{0.40, 0.26, 0.18};
    // light scattered inside the petal: pale roses glow, their shadows stay light
    double scatter     = 0.0;
};
static PetalPalette PC;                    // working copy: the defaults above (or a preset) + variation

// Sample the palette gradient at t in [0,1] with smooth interpolation between stops.
static V3 paletteSample(double t){
    const auto& s = PC.stops;
    if(s.empty()) return V3(1,0,0);
    t = clampd(t,0.0,1.0);
    if(t<=s.front().pos) return s.front().rgb;
    if(t>=s.back().pos)  return s.back().rgb;
    for(size_t i=0;i+1<s.size();++i){
        if(t>=s[i].pos && t<=s[i+1].pos){
            double span = s[i+1].pos - s[i].pos;
            double f = span>1e-9 ? (t - s[i].pos)/span : 0.0;
            f = f*f*(3.0-2.0*f);                       // smoothstep -> gentle blend
            return mix(s[i].rgb, s[i+1].rgb, f);
        }
    }
    return s.back().rgb;
}

// Per-vertex front-face petal colour from the palette.
//   xRadial : 0 = throat, 1 = petal rim
//   depth   : 0 = outer petal, 1 = inner petal of the bloom
static V3 petalColor(double xRadial, double depth){
    double wr=PC.axisRadial, wd=PC.axisDepth, ws=wr+wd;
    if(ws<1e-9){ wr=1.0; wd=0.0; ws=1.0; }
    double t = (wr*xRadial + wd*depth)/ws;
    if(PC.bias>0 && std::fabs(PC.bias-1.0)>1e-6) t = std::pow(clampd(t,0,1), PC.bias);
    V3 c = paletteSample(t);
    if(PC.edgeEnable){
        double lo = clampd(1.0 - PC.edgeWidth, 0.0, 1.0);
        double e  = smoothstep(lo, 1.0, xRadial);
        c = mix(c, PC.edgeColor, e);
    }
    if(PC.throatAmt>0) c = mix(c, PC.throatColor, PC.throatAmt*(1.0-smoothstep(0.0,0.28,xRadial)));
    if(PC.ageAmt>0)    c = mix(c, PC.ageColor, PC.ageAmt*std::pow(1.0-depth,3.0)*smoothstep(0.72,1.0,xRadial));
    return c;
}

struct RoseFamily { const char* name; std::vector<ColorStop> stops; double bias; V3 throat; double scatter; };
static const std::vector<RoseFamily> ROSE_FAMILIES = {
    {"crimson",        {{0.0,{0.42,0.02,0.09}},{1.0,{0.86,0.10,0.20}}},                         0.7, {0.95,0.80,0.30}, 0.00},
    {"velvet red",     {{0.0,{0.28,0.01,0.05}},{1.0,{0.68,0.03,0.10}}},                         0.8, {0.90,0.70,0.25}, 0.00},
    {"scarlet",        {{0.0,{0.55,0.04,0.05}},{1.0,{0.92,0.16,0.12}}},                         0.7, {1.00,0.80,0.25}, 0.05},
    {"pink",           {{0.0,{0.78,0.30,0.42}},{1.0,{0.98,0.62,0.72}}},                         0.8, {1.00,0.90,0.60}, 0.30},
    {"fuchsia",        {{0.0,{0.55,0.04,0.28}},{1.0,{0.90,0.22,0.55}}},                         0.8, {1.00,0.85,0.50}, 0.10},
    {"blush white",    {{0.0,{0.96,0.90,0.78}},{1.0,{1.00,0.93,0.92}}},                         1.0, {0.95,0.95,0.60}, 0.50},
    {"yellow",         {{0.0,{0.98,0.72,0.12}},{1.0,{1.00,0.90,0.42}}},                         0.9, {1.00,0.65,0.10}, 0.45},
    {"apricot",        {{0.0,{0.96,0.55,0.25}},{1.0,{1.00,0.80,0.62}}},                         0.9, {1.00,0.75,0.30}, 0.40},
    {"coral",          {{0.0,{0.92,0.32,0.18}},{1.0,{0.99,0.55,0.42}}},                         0.8, {1.00,0.80,0.30}, 0.25},
    {"sunset bicolour",{{0.0,{0.97,0.82,0.25}},{0.6,{0.96,0.42,0.18}},{1.0,{0.78,0.06,0.10}}}, 0.8, {1.00,0.85,0.30}, 0.25},
    {"mauve",          {{0.0,{0.62,0.45,0.60}},{1.0,{0.84,0.70,0.84}}},                         0.9, {0.95,0.90,0.70}, 0.35},
};
static bool        g_pickFamily = true;    // false keeps the palette's colours (e.g. a chosen preset)
static const char* g_familyName = "palette";

static void randomizePetalColors(){
    static const PetalPalette basePC = PC;           // captured on the first call: defaults or preset
    PC = basePC;
    g_familyName = "palette";
    if(g_pickFamily && RANDOMNESS>0 && rnd(0.0,1.0) > 0.35){
        const RoseFamily& f = ROSE_FAMILIES[(size_t)rnd(0.0,(double)ROSE_FAMILIES.size()) % ROSE_FAMILIES.size()];
        PC.stops=f.stops; PC.bias=f.bias; PC.throatColor=f.throat; PC.scatter=f.scatter; g_familyName=f.name;
    }
    // nudge the whole palette, and each stop a touch on its own
    const double dh=RANDOMNESS*rnd(-5.0,5.0), sm=vary(1.0,0.08), vm=vary(1.0,0.06);
    for(ColorStop& cs : PC.stops) cs.rgb = hsvAdjust(cs.rgb, dh + RANDOMNESS*rnd(-2.0,2.0), sm, vm);
    PC.edgeColor    = hsvAdjust(PC.edgeColor,    dh, sm, vm);
    PC.reverseColor = hsvAdjust(PC.reverseColor, dh, sm, vm);
    PC.bias      = vary(PC.bias, 0.20);
    PC.scatter   = vary(PC.scatter, 0.20);
    PC.axisDepth = PC.axisDepth + RANDOMNESS*rnd(0.0,0.30);   // inner petals a shade apart
    PC.throatAmt = PC.throatAmt + RANDOMNESS*rnd(0.0,0.35);
    PC.ageAmt    = PC.ageAmt    + RANDOMNESS*rnd(0.0,0.35);
    if(!PC.edgeEnable && rnd(0.0,1.0) < 0.10*RANDOMNESS){      // picotee: a darker, richer rim
        PC.edgeEnable=true;
        PC.edgeColor = hsvAdjust(PC.stops.back().rgb, -6.0, 1.35, 0.80);
        PC.edgeWidth = rnd(0.12,0.25);
    }
}


static void buildRose(){
    const size_t first = g_tris.size();
    // petals = petalFreq * thMax / 2π  ->  solve for the winding length we need
    const double thMax = RP.numPetals * 2.0*PI / RP.petalFreq;
    const int    NT = std::max(8, RP.numPetals * RP.resAround); // samples around the spiral
    const int    NX = std::max(2, RP.resAcross);                // samples across each petal strip
    const double e = 1e-3;

    auto P = [](double th,double x){ return rosePoint(th,x); };
    auto N = [&](double th,double x){
        V3 dt = P(th+e,x) - P(th-e,x);
        V3 dx = P(th,clampd(x+e,0,1)) - P(th,clampd(x-e,0,1));
        return norm(cross(dt,dx));
    };

    std::vector<V3> pos((NT+1)*(NX+1)), nrm((NT+1)*(NX+1));
    std::vector<V3> col((NT+1)*(NX+1));
    std::vector<double> uu((NT+1)*(NX+1)), vv((NT+1)*(NX+1));
    auto idx=[&](int i,int j){ return i*(NX+1)+j; };
    for(int i=0;i<=NT;++i){
        double th = thMax * (double)i/NT;
        double depth = (double)i/NT;                                // 0 outer -> 1 inner
        for(int j=0;j<=NX;++j){
            double x = (double)j/NX;
            pos[idx(i,j)] = P(th,x);
            nrm[idx(i,j)] = N(th,x);
            const PetalVar pv = petalVarAt(th);
            col[idx(i,j)] = hsvAdjust(petalColor(x, depth)*pv.shade, pv.hue, 1.0, 1.0);
            uu[idx(i,j)] = x;                                          // base(0) -> tip(1)
            vv[idx(i,j)] = std::fmod(RP.petalFreq*th, 2.0*PI)/(2.0*PI); // lateral within a petal
        }
    }
    // ambient occlusion, baked per vertex. The bloom is one spiral, so at any
    // angle around the axis it passes again one turn further out and one turn
    // further in (and two, three...). In that vertical slice through the axis
    // those neighbouring turns form a horizon on either side of each point:
    // the higher they rise, the less of the sky the point can see.
    std::vector<double> occ((NT+1)*(NX+1), 1.0);
    for(int i=0;i<=NT;++i){
        double th = thMax * (double)i/NT;
        for(int j=0;j<=NX;++j){
            V3 p=pos[idx(i,j)];
            double rp=std::hypot(p.x,p.z), zp=p.y, hOut=0, hIn=0;
            for(int k=-3;k<=3;++k){
                double t2=th+2.0*PI*k;
                if(k==0 || t2<0 || t2>thMax) continue;
                for(int m=0;m<8;++m){
                    V3 q=rosePoint(t2,(m+0.5)/8.0);
                    double dr=std::hypot(q.x,q.z)-rp, dz=q.y-zp;
                    if(dr>0) hOut=std::max(hOut,std::atan2(dz, dr));   // horizon outwards
                    else     hIn =std::max(hIn, std::atan2(dz,-dr));   // horizon inwards
                }
            }
            occ[idx(i,j)] = clampd((PI - hOut - hIn)/PI, 0.0, 1.0);
        }
    }
    // smooth it over the surface (a few 3x3 box passes): occlusion is soft light
    for(int pass=0;pass<2;++pass){
        std::vector<double> sm(occ.size());
        for(int i=0;i<=NT;++i) for(int j=0;j<=NX;++j){
            double acc=0; int n=0;
            for(int di=-1;di<=1;++di) for(int dj=-1;dj<=1;++dj){
                int a=i+di, b=j+dj;
                if(a<0||a>NT||b<0||b>NX) continue;
                acc+=occ[idx(a,b)]; ++n;
            }
            sm[idx(i,j)]=acc/n;
        }
        occ.swap(sm);
    }
    auto MV=[&](int i,int j){ return Vert{pos[idx(i,j)],nrm[idx(i,j)],col[idx(i,j)],uu[idx(i,j)],vv[idx(i,j)],
                                          (double)i/NT,(double)j/NX,occ[idx(i,j)]}; };
    for(int i=0;i<NT;++i) for(int j=0;j<NX;++j){
        Vert v00=MV(i,j), v10=MV(i+1,j), v01=MV(i,j+1), v11=MV(i+1,j+1);
        g_tris.push_back({v00,v10,v11,0,(double)NT,(double)NX});
        g_tris.push_back({v00,v11,v01,0,(double)NT,(double)NX});
    }
    tagNewMesh(first);
    // bounding box of the bloom (for camera auto-framing)
    g_bloomMin = V3( 1e30, 1e30, 1e30);
    g_bloomMax = V3(-1e30,-1e30,-1e30);
    for(const V3& p : pos){
        g_bloomMin.x=std::min(g_bloomMin.x,p.x); g_bloomMax.x=std::max(g_bloomMax.x,p.x);
        g_bloomMin.y=std::min(g_bloomMin.y,p.y); g_bloomMax.y=std::max(g_bloomMax.y,p.y);
        g_bloomMin.z=std::min(g_bloomMin.z,p.z); g_bloomMax.z=std::max(g_bloomMax.z,p.z);
    }
}

// ----------------------------------------------------------------------------
// tubes: the stem and the leaf stalks
// ----------------------------------------------------------------------------
template<class Axis, class Radius, class Colour>
static void buildTube(Axis axis, Radius radius, int RINGS, int SEG, Colour colAt){  // t: 0..1 along the axis
    const size_t first = g_tris.size();
    // SEG+1 columns: the last one repeats the first, so the grid coordinate
    // runs 0..1 around the tube instead of jumping back to 0 at the seam
    std::vector<std::vector<Vert>> ring(RINGS+1, std::vector<Vert>(SEG+1));
    for(int i=0;i<=RINGS;++i){
        double t=(double)i/RINGS;
        V3 c=axis(t);
        V3 tangent=norm(axis(std::min(1.0,t+1e-3))-axis(std::max(0.0,t-1e-3)));
        V3 up = std::fabs(tangent.y)<0.9? V3(0,1,0):V3(1,0,0);
        V3 u=norm(cross(tangent,up)), w=norm(cross(tangent,u));
        double rad=radius(t);
        for(int s=0;s<=SEG;++s){
            double a=2.0*PI*s/SEG;
            V3 nrml=norm(u*std::cos(a)+w*std::sin(a));
            ring[i][s]={ c+nrml*rad, nrml, colAt(t), 0, 0, t, (double)s/SEG };
        }
    }
    for(int i=0;i<RINGS;++i) for(int s=0;s<SEG;++s){
        int s2=s+1;
        g_tris.push_back({ring[i][s],ring[i+1][s],ring[i+1][s2],1,(double)RINGS,(double)SEG});
        g_tris.push_back({ring[i][s],ring[i+1][s2],ring[i][s2],1,(double)RINGS,(double)SEG});
    }
    tagNewMesh(first);
}

// stem: a gently curved, tapered tube.  t: 0 top -> 1 bottom
struct StemShape {
    double bendX=0.10, freqX=1.1, bendZ=0.04, freqZ=1.7;  // the gentle S-bend
    double wiggle=0.0, freqW=4.0;                         // small extra kinks
    double radius=0.045, taper=0.45;
};
static StemShape g_stem;
static double g_stemYTop=0.0, g_stemYBot=-2.6;
static V3 stemAxis(double t){                      // every term is 0 at t=0: the top stays under the bloom
    double y = g_stemYTop + (g_stemYBot-g_stemYTop)*t;
    double x = g_stem.bendX*std::sin(t*PI*g_stem.freqX) + g_stem.wiggle*std::sin(t*PI*g_stem.freqW);
    double z = g_stem.bendZ*std::sin(t*PI*g_stem.freqZ) + 0.7*g_stem.wiggle*std::sin(t*PI*g_stem.freqW*1.37);
    return V3(x,y,z);
}

static void randomizeStem(){
    g_stem = StemShape{};
    g_stem.bendX  = g_stem.bendX + RANDOMNESS*rnd(-0.07,0.07);
    g_stem.freqX  = vary(g_stem.freqX, 0.25);
    g_stem.bendZ  = g_stem.bendZ + RANDOMNESS*rnd(-0.05,0.05);
    g_stem.freqZ  = vary(g_stem.freqZ, 0.25);
    g_stem.wiggle = RANDOMNESS*rnd(0.0,0.018);
    g_stem.freqW  = rnd(3.0,6.0);
    g_stem.radius = vary(g_stem.radius, 0.12);
    g_stem.taper  = vary(g_stem.taper,  0.20);
}

static const V3 STEM_GREEN  {0.16,0.40,0.13};
static const V3 YOUNG_BRONZE{0.42,0.20,0.10};      // new growth on a rose is flushed red-bronze

static inline double stemRadius(double t){ return g_stem.radius*(1.0 - g_stem.taper*t); }

static void buildStem(double yTop,double yBottom){
    g_stemYTop=yTop; g_stemYBot=yBottom;
    buildTube(stemAxis, stemRadius, 40, 12, [](double t){       // young (reddish) just under the bloom
        return mix(STEM_GREEN, YOUNG_BRONZE, 0.55*(1.0-smoothstep(0.0,0.30,t))); });
}

// prickles ("thorns"): broad-based, tapering, hooked a little towards the
// ground; reddish when young near the top, tan and woody further down
static void buildThorns(){
    const int n = 9 + (int)rnd(0.0, 8.0);
    for(int k=0;k<n;++k){
        const double t=rnd(0.05,0.92), a=rnd(0.0,2.0*PI);
        const V3 c=stemAxis(t);
        const V3 down=norm(stemAxis(std::min(1.0,t+1e-3))-stemAxis(std::max(0.0,t-1e-3)));  // along the stem
        V3 out(std::cos(a),0,std::sin(a)); out=norm(out - down*dot(out,down));
        const double R=stemRadius(t), L=R*vary(2.3,0.25), r0=R*vary(0.70,0.20);
        const double young=1.0-smoothstep(0.0,0.35,t);
        const V3 cBase=mix(V3(0.28,0.30,0.12), V3(0.45,0.20,0.10), young);
        const V3 cTip =mix(V3(0.62,0.50,0.32), V3(0.70,0.30,0.18), young);
        const V3 b0=c + out*(0.5*R);                              // rooted inside the stem
        buildTube([&](double s){ return b0 + out*((L+0.5*R)*s) + down*(0.45*L*s*s); },
                  [&](double s){ return r0*std::pow(1.0-s,1.2) + 1e-4; }, 6, 8,
                  [&](double s){ return mix(cBase,cTip,s); });
    }
}

struct LeafLook {
    int    teeth      = 18;     // serration teeth along each side of a leaflet
    double toothDepth = 0.11;   // notch depth, as a fraction of the local half-width
    int    veins      = 8;      // secondary veins on each side of the midrib
    double veinDepth  = 1.0;    // how deeply the veins are pressed in (bump strength)
    double speckles   = 1.0;    // amount of dark specks & pale flecks (0 = spotless)
    double bronze     = 0.35;   // reddish tint on margins and tips, as on young rose leaves
    V3     top   = V3{0.07, 0.21, 0.05};   // upper surface: deep, glossy green
    V3     under = V3{0.25, 0.37, 0.20};   // underside: paler, a little grey
    V3     vein  = V3{0.30, 0.46, 0.15};   // midrib & veins seen from above
    V3     edge  = V3{0.40, 0.12, 0.05};   // bronze-red margin
};
static LeafLook LL;

// leaflet half-width profile along its length u in [0,1], peak = 1
static inline double leafEnv(double u){
    static const double um=0.5/1.3, k=1.0/(std::sqrt(um)*std::pow(1.0-um,0.8));
    u=clampd(u,0.0,1.0);
    return k*std::sqrt(u)*std::pow(1.0-u,0.8);
}

// how far across (|v|, 1 = the envelope) the serrated margin reaches at this spot;
// teeth lean towards the tip and the two sides are staggered
static inline double leafLimit(double u,double v){
    double saw = u*LL.teeth + (v<0 ? 0.5 : 0.0);
    saw -= std::floor(saw);
    double fade = smoothstep(0.06,0.22,u)*(1.0-smoothstep(0.93,1.0,u));   // smooth base & tip
    return 1.0 - LL.toothDepth*fade*std::pow(1.0-saw,1.3);
}

// one leaflet, built in a local frame: dir = along the midrib, upRef ~ blade normal
static void buildLeaflet(V3 base, V3 dir, V3 upRef, double length, double width,
                         double fold, double droop, V3 topCol){
    const size_t first = g_tris.size();
    V3 f=norm(dir);
    V3 s=norm(cross(f,upRef));
    V3 n=norm(cross(s,f));                             // upper-surface normal
    const int NU=28, NV=12;
    auto P=[&](double u,double v){
        double env=leafEnv(u);
        double lift = fold*width*std::pow(std::fabs(v),1.6)*env        // shallow V along the midrib
                    - droop*length*u*u                                 // tip arches down
                    + 0.012*length*std::sin(u*PI*5.0)*std::fabs(v);   // slight waviness
        return base + f*(u*length) + s*(v*0.5*width*env) + n*lift;
    };
    std::vector<std::vector<Vert>> g(NU+1,std::vector<Vert>(NV+1));
    for(int i=0;i<=NU;++i) for(int j=0;j<=NV;++j){
        double u=(double)i/NU, v=((double)j/NV)*2.0-1.0;
        const double e=1e-3;
        V3 du=P(clampd(u+e,0,1),v)-P(clampd(u-e,0,1),v);
        V3 dv=P(u,v+e)-P(u,v-e);
        V3 c=cross(dv,du);                              // points to the upper side
        g[i][j]={P(u,v), len(c)>1e-12? norm(c) : n, topCol, u, v, u, (double)j/NV};
    }
    for(int i=0;i<NU;++i) for(int j=0;j<NV;++j){
        g_tris.push_back({g[i][j],g[i+1][j],g[i+1][j+1],2,(double)NU,(double)NV});
        g_tris.push_back({g[i][j],g[i+1][j+1],g[i][j+1],2,(double)NU,(double)NV});
    }
    tagNewMesh(first);
}

static void buildRoseLeaf(double tStem, double azimuth, double scale, double lift, int pairs, double youth){
    const V3 P0=stemAxis(tStem), up(0,1,0);
    const V3 out(std::cos(azimuth),0,std::sin(azimuth)), across(-out.z,0,out.x);
    const double L=0.75*scale;
    const double bend=RANDOMNESS*rnd(-0.10,0.10);      // the stalk curves a little sideways
    auto rachis=[&](double t){ return P0 + out*(L*t) + up*(L*(lift*t - 0.45*t*t)) + across*(L*bend*t*t); };
    const V3 stalkCol=mix(V3{0.15,0.36,0.11}, YOUNG_BRONZE, 0.5*youth);
    const V3 topCol  =mix(LL.top, V3{0.26,0.10,0.05}, 0.55*youth);
    buildTube(rachis, [&](double t){ return 0.014*scale*(1.0-0.55*t); }, 18, 6, [&](double){ return stalkCol; });
    auto frameAt=[&](double t, V3& tg, V3& side, V3& n0){
        tg  =norm(rachis(std::min(1.0,t+1e-3))-rachis(std::max(0.0,t-1e-3)));
        side=norm(cross(tg,up));
        n0  =norm(cross(side,tg));
    };
    // stipules: a pair of narrow, fringed wings hugging the base of the stalk
    for(int sg=-1;sg<=1;sg+=2){
        V3 tg,side,n0; frameAt(0.03,tg,side,n0);
        buildLeaflet(rachis(0.02) + side*(sg*0.012*scale), norm(tg + side*(sg*0.30) + n0*0.10), n0,
                     0.20*scale, 0.30*0.20*scale, 0.05, 0.0, topCol);
    }
    // opposite pairs of side leaflets, growing a little larger towards the tip...
    const double tLo  = pairs>=3 ? 0.30 : 0.40, tHi = pairs>=3 ? 0.74 : 0.70;
    const double lenLo= pairs>=3 ? 0.34 : 0.40;
    for(int k=0;k<pairs;++k) for(int sg=-1;sg<=1;sg+=2){
        double fk = pairs>1 ? (double)k/(pairs-1) : 0.5;
        double tk = tLo + (tHi-tLo)*fk;
        V3 tg,side,n0; frameAt(tk,tg,side,n0);
        double ang=(62.0 - 8.0*fk + RANDOMNESS*rnd(-7.0,7.0))*PI/180.0;
        V3 dir =norm(tg*std::cos(ang) + side*(sg*std::sin(ang)) - n0*(0.06 + RANDOMNESS*rnd(-0.05,0.08)));
        V3 upR =norm(n0 + side*(sg*(0.15 + RANDOMNESS*rnd(-0.08,0.08))));  // blades slope away from the stalk
        double l=(lenLo + (0.47-lenLo)*fk)*scale*vary(1.0,0.10);
        buildLeaflet(rachis(tk), dir, upR, l, vary(0.60,0.08)*l, vary(0.10,0.3), vary(0.08,0.5), topCol);
    }
    // ...and the terminal leaflet, the largest
    V3 tg,side,n0; frameAt(1.0,tg,side,n0);
    double l=0.55*scale*vary(1.0,0.08);
    buildLeaflet(rachis(1.0), norm(tg - n0*0.05 + side*(RANDOMNESS*rnd(-0.12,0.12))), n0,
                 l, vary(0.60,0.08)*l, vary(0.10,0.3), vary(0.10,0.5), topCol);
}

static void buildLeaf(V3 base, V3 dir, V3 sideRef, double length, double width,
                      double fold, V3 colMid, V3 colEdge){
    const size_t first = g_tris.size();
    V3 f=norm(dir);
    V3 s=norm(cross(f,sideRef)); if(len(s)<1e-6) s=V3(1,0,0);
    V3 up=norm(cross(s,f));
    const int NU=16, NV=6;
    auto half=[&](double u){ return std::sin(PI*std::pow(u,0.8)); };  // leaf outline
    std::vector<std::vector<Vert>> g(NU+1,std::vector<Vert>(NV+1));
    for(int i=0;i<=NU;++i){
        double u=(double)i/NU;
        for(int j=0;j<=NV;++j){
            double v=((double)j/NV)*2.0-1.0;            // -1..1 across the leaf
            double hw=half(u)*0.5*width;
            double lat=v*hw;
            double lift=fold*(1.0-std::fabs(v))*half(u); // fold up along the midrib
            V3 p = base + f*(u*length) + s*lat + up*lift;
            V3 col = colMid + (colEdge-colMid)*std::fabs(v);
            g[i][j]={p,V3(0,0,0),col,0,0,u,(double)j/NV};
        }
    }
    for(int i=0;i<NU;++i) for(int j=0;j<NV;++j){
        Vert a=g[i][j],b=g[i+1][j],c=g[i+1][j+1];
        V3 n1=norm(cross(b.p-a.p,c.p-a.p));
        a.n=b.n=c.n=n1; g_tris.push_back({a,b,c,1,(double)NU,(double)NV});
        Vert a2=g[i][j],c2=g[i+1][j+1],d2=g[i][j+1];
        V3 n2=norm(cross(c2.p-a2.p,d2.p-a2.p));
        a2.n=c2.n=d2.n=n2; g_tris.push_back({a2,c2,d2,1,(double)NU,(double)NV});
    }
    tagNewMesh(first);
}

// sepals: slim green leaves curling up around the base of the bloom
static void buildSepals(V3 base){
    V3 mid{0.20,0.45,0.16}, edge{0.12,0.34,0.10};
    int n=5;
    const double a0=0.4 + RANDOMNESS*rnd(-PI/5,PI/5);
    for(int k=0;k<n;++k){
        double a=2.0*PI*k/n + a0 + RANDOMNESS*rnd(-0.2,0.2);
        double o=vary(0.7,0.2);                                  // how far it leans out
        V3 dir=norm(V3(std::cos(a)*o, 0.9, std::sin(a)*o));       // outward + upward
        V3 sideRef(-std::sin(a),0,std::cos(a));
        buildLeaf(base, dir, sideRef, vary(0.55,0.15), vary(0.16,0.15), 0.06, mid, edge);
    }
}

struct PetalLook {
    double veinDepth = 0.11;   // strength of the base->tip veins (0 = perfectly smooth)
    int    veinCount = 11;     // number of veins across each petal
    double micro     = 0.025;  // fine micro-texture amount
    double sheen     = 0.16;   // broad satin highlight (low = matte velvet)
    double rim       = 0.55;   // velvety lighter edge (Fresnel rim)
    double backlight = 0.22;   // warm glow of light passing through the thin petal
    double veinTint  = 0.0;    // how much darker the vein grooves are
    double mottle    = 0.05;   // broad, soft colour mottling ...
    double mottle2   = 0.03;   // ... and a finer band of it
    double mottlePhase = 0.0;
    double occlusion = 0.9;    // strength of the shading deep inside the bloom (0 = none)
};
static PetalLook PL;                       // working copy: the defaults above + random variation

struct Lighting {
    V3     keyDir  = V3( 0.35, 0.65, 0.75);  V3 keyCol  = V3(1.00, 0.94, 0.86)*1.35;
    V3     fillDir = V3(-0.80, 0.10, 0.55);  V3 fillCol = V3(0.55, 0.66, 0.90)*0.30;
    V3     backDir = V3(-0.30, 0.40,-0.87);  V3 backCol = V3(1.00, 0.88, 0.76)*1.50;
    V3     ambient = V3(0.10, 0.11, 0.14);   // soft sky light (reduced by occlusion)
    double exposure = 0.80;
};
static Lighting LIGHT;                     // working copy: the defaults above + random variation
static V3 g_Lkey, g_Lfill, g_Lback;       // normalised light directions

static inline V3 toLinear (V3 c){ return mul(c,c); }                    // display -> linear (gamma 2)
static inline V3 toDisplay(V3 c){ return { std::sqrt(std::max(0.0,c.x)), std::sqrt(std::max(0.0,c.y)),
                                            std::sqrt(std::max(0.0,c.z)) }; }
static inline double aces(double x){                                    // filmic curve (Narkowicz fit)
    x=std::max(0.0,x);
    return clampd((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0);
}
// a richer, more saturated version of a (linear) colour with the same peak:
// what the shadowed side of a coloured petal turns to — crimson, not grey
static inline V3 deepen(V3 c){
    double m=std::max({c.x,c.y,c.z,1e-4});
    return mul(c,c)*(1.0/m);
}
static inline double softDiffuse(V3 N, V3 L){                          // half-Lambert, squared
    double d=clampd(dot(N,L)*0.5+0.5,0.0,1.0); return d*d;
}
static inline double wrapDiffuse(V3 N, V3 L, double w){                // Lambert wrapped a little
    double d=clampd((dot(N,L)+w)/(1.0+w),0.0,1.0); return d*d;         // past the terminator
}

static void randomizeLighting(){
    static const PetalLook basePL = PL;              // captured on the first call
    PL = basePL;
    PL.veinDepth = vary(PL.veinDepth, 0.35);
    PL.veinCount = std::max(5, PL.veinCount + (int)std::lround(RANDOMNESS*rnd(-3.0,3.0)));
    PL.micro     = vary(PL.micro,     0.40);
    PL.sheen     = vary(PL.sheen,     0.40);
    PL.rim       = vary(PL.rim,       0.30);
    PL.backlight = vary(PL.backlight, 0.35);
    PL.veinTint  = PL.veinTint + RANDOMNESS*rnd(0.0,0.10);
    PL.mottle    = vary(PL.mottle,    0.50);
    PL.mottle2   = vary(PL.mottle2,   0.50);
    PL.mottlePhase = PL.mottlePhase + RANDOMNESS*rnd(0.0,2.0*PI);
    PL.occlusion = clampd(vary(PL.occlusion, 0.20), 0.0, 1.0);

    LIGHT = Lighting{};
    auto turn=[](V3 d,double yaw,double pitch){
        d=V3( std::cos(yaw)*d.x + std::sin(yaw)*d.z, d.y, -std::sin(yaw)*d.x + std::cos(yaw)*d.z);
        return V3( d.x, std::cos(pitch)*d.y - std::sin(pitch)*d.z, std::sin(pitch)*d.y + std::cos(pitch)*d.z);
    };
    LIGHT.keyDir  = turn(LIGHT.keyDir,  RANDOMNESS*rnd(-25.0,25.0)*PI/180.0, RANDOMNESS*rnd(-10.0,10.0)*PI/180.0);
    LIGHT.backDir = turn(LIGHT.backDir, RANDOMNESS*rnd(-30.0,30.0)*PI/180.0, RANDOMNESS*rnd(-10.0,10.0)*PI/180.0);
    const double warm=RANDOMNESS*rnd(-1.0,1.0);                // cool morning ... warm evening
    LIGHT.keyCol  = mul(LIGHT.keyCol, V3(1.0+0.06*warm, 1.0+0.01*warm, 1.0-0.07*warm))*vary(1.0,0.08);
    LIGHT.fillCol = LIGHT.fillCol*vary(1.0,0.30);
    LIGHT.backCol = LIGHT.backCol*vary(1.0,0.30);
    LIGHT.exposure= vary(LIGHT.exposure,0.05);
    g_Lkey=norm(LIGHT.keyDir); g_Lfill=norm(LIGHT.fillDir); g_Lback=norm(LIGHT.backDir);
}

static int WIN_W=820, WIN_H=900;          // window / output size
#ifdef HEADLESS
static int SS=2;                          // supersampling for crisp stills
#else
static int SS=1;                          // realtime; raise to 2 for nicer (slower) frames
#endif
static int RW, RH;                        // internal render (supersampled) size
static std::vector<uint32_t> g_fb;        // render buffer (RW*RH)
static std::vector<float>    g_zb;        // depth buffer (RW*RH)
static std::vector<uint32_t> g_out;       // downsampled output (WIN_W*WIN_H)
static std::vector<float>    g_wm, g_tmp; // wireframe line mask (RW*RH) and blur scratch

static void makePlainBackground();
static void setupBuffers(){
    RW=WIN_W*SS; RH=WIN_H*SS;
    g_fb.assign((size_t)RW*RH,0); g_zb.assign((size_t)RW*RH,1e30f);
    g_out.assign((size_t)WIN_W*WIN_H,0);
    g_wm.assign((size_t)RW*RH,0.0f); g_tmp.assign((size_t)RW*RH,0.0f);
    makePlainBackground();
}

static inline uint32_t pack(V3 c){
    int r=(int)clampd(c.x*255.0+0.5,0,255);
    int g=(int)clampd(c.y*255.0+0.5,0,255);
    int b=(int)clampd(c.z*255.0+0.5,0,255);
    return (0xFFu<<24)|(r<<16)|(g<<8)|b;
}

// ---- rose leaf surface: veins, speckles, gloss ----
struct LeafVeins { double mid, sec, ter; };        // midrib, secondary veins, veinlet net (0..1)
static LeafVeins leafVeins(double u,double v){
    double a=std::fabs(v)*leafEnv(u);                 // distance from the midrib (1 = widest point)
    LeafVeins r;
    r.mid = 1.0 - smoothstep(0.0, 0.030*(1.0-0.6*u)+0.006, a);
    double sv = (u - 0.30*a - 0.20*a*a*a)*LL.veins + (v<0 ? 0.5 : 0.0);  // arching towards the tip
    double dv = std::fabs(sv - std::round(sv));
    r.sec = (1.0 - smoothstep(0.0, 0.09, dv))
          * smoothstep(0.03,0.08,a) * (1.0-smoothstep(0.80,0.97,std::fabs(v)))
          * smoothstep(0.03,0.12,u) * (1.0-smoothstep(0.88,1.0,u));
    double t1 = std::fabs(std::sin(PI*(u*46.0 + a*13.0)));
    double t2 = std::fabs(std::sin(PI*(a*28.0 - u*9.0)));
    r.ter = (1.0 - smoothstep(0.0, 0.14, std::min(t1,t2))) * (1.0-smoothstep(0.85,1.0,std::fabs(v)));
    return r;
}
static inline double leafHeight(double u,double v){  // veins sit in grooves of the upper surface
    LeafVeins w=leafVeins(u,v);
    return -(1.0*w.mid + 0.7*w.sec + 0.12*w.ter);
}

// dark specks with a yellowish halo, tiny pale flecks and a soft mottle, scattered
// with a hash on a jittered grid over the leaflet (different on every leaflet)
static V3 leafSpeckle(V3 albedo, double u, double v, uint64_t seed){
    double as=v*leafEnv(u)*0.30;                      // across, in leaflet lengths
    const double K=45.0;
    double cu=u*K, cv=as*K;
    int iu=(int)std::floor(cu), iv=(int)std::floor(cv);
    double dark=0, halo=0, pale=0;
    for(int du=-1;du<=1;++du) for(int dv=-1;dv<=1;++dv){
        uint64_t h=hash64(seed ^ hash64(((uint64_t)(int64_t)(iu+du)<<32) ^ (uint64_t)(uint32_t)(iv+dv)));
        double p=hashUnit(h,0);
        double d=std::hypot(cu-(iu+du+0.2+0.6*hashUnit(h,1)), cv-(iv+dv+0.2+0.6*hashUnit(h,2)));
        if(p < 0.035*LL.speckles){
            double r=0.12+0.22*hashUnit(h,3);
            dark=std::max(dark, 1.0-smoothstep(0.6*r, r, d));
            halo=std::max(halo, 1.0-smoothstep(r, 2.4*r, d));
        } else if(p < 0.16*LL.speckles){
            double r=0.08+0.08*hashUnit(h,3);
            pale=std::max(pale, 1.0-smoothstep(0.4*r, r, d));
        }
    }
    double ph=6.28*hashUnit(seed,1);
    albedo = albedo*(1.0 + 0.10*LL.speckles*std::sin(u*11.0+ph)*std::sin(as*37.0+ph*1.7));
    albedo = mix(albedo, V3(0.50,0.48,0.10), 0.35*std::max(0.0, halo-dark));
    albedo = mix(albedo, albedo*1.5 + V3(0.03,0.04,0.0), 0.5*pale);
    albedo = mix(albedo, V3(0.07,0.05,0.03), 0.85*dark);
    return albedo;
}

static V3 shadeLeaf(V3 Pc, V3 Nin, V3 base, double u, double v, V3 Tn, V3 Bn, bool hasTB, uint64_t seed){
    V3 toCam=norm(Pc*-1.0);
    V3 N=norm(Nin);
    bool back = dot(N,toCam) < 0;                      // looking at the underside
    if(hasTB){                                         // bump: the vein grooves
        const double e=1e-3;
        double hu=(leafHeight(u+e,v)-leafHeight(u-e,v))/(2*e);
        double hv=(leafHeight(u,v+e)-leafHeight(u,v-e))/(2*e);
        double kv=0.012/std::max(leafEnv(u),0.3);      // v spans less of the blade near base & tip
        N=norm(N - (Tn*(hu*0.006) + Bn*(hv*kv))*LL.veinDepth);
    }
    if(back) N=N*-1.0;

    double var=hashUnit(hash64(seed),0);               // per-leaflet variation
    LeafVeins w=leafVeins(u,v);
    V3 albedo = (back ? LL.under : base)*(0.88+0.24*var);
    double vm = back ? 0.85*w.mid + 0.55*w.sec + 0.12*w.ter
                     : 0.75*w.mid + 0.45*w.sec + 0.05*w.ter;
    albedo = mix(albedo, back ? V3(0.48,0.56,0.34) : LL.vein, clampd(vm,0,1));
    double br = LL.bronze*(0.5+var)*clampd(smoothstep(0.78,1.0,std::fabs(v)) + 0.7*smoothstep(0.80,1.0,u),0,1);
    albedo = mix(albedo, LL.edge, clampd(br*(back?0.4:1.0),0,1));
    albedo = leafSpeckle(albedo,u,v,seed);

    // glossy upper side, matte underside, light glowing through the thin blade
    const V3 alb=toLinear(albedo), deep=deepen(alb);
    double dK=wrapDiffuse(N,g_Lkey,0.35), dF=softDiffuse(N,g_Lfill);
    double nh=clampd(dot(N,norm(g_Lkey+toCam)),0,1);
    double spec = back ? 0.04*std::pow(nh,12.0) : 0.45*std::pow(nh,60.0) + 0.08*std::pow(nh,12.0);
    double fres = std::pow(1.0-clampd(dot(N,toCam),0,1),3.0);
    double thru = clampd(-dot(N,g_Lback),0,1);                     // the back light, through the blade
    double fwd  = clampd(-dot(toCam,g_Lback),0,1); fwd=fwd*fwd*fwd;
    V3 transC = mix(alb, V3(0.20,0.45,0.02), 0.5);                  // glows yellow-green
    return mul(mix(deep,alb,dK)*dK, LIGHT.keyCol)
         + mul(alb*dF, LIGHT.fillCol)
         + mul(alb, LIGHT.ambient)
         + mul(transC*(0.45*thru*(0.3+0.7*fwd)), LIGHT.backCol)
         + mul(alb*(fres*0.25*(0.3+0.7*softDiffuse(N,g_Lback))), LIGHT.backCol)
         + mul(V3(0.9,1.0,0.85)*spec, LIGHT.keyCol);
}

// per-pixel shading. Pc/Nin in view space; Tn,Bn = surface tangent frame (petals, leaves).
// linear-light shading of a surface point; Pc/Nin in view space, Tn,Bn = surface tangent frame
static V3 shadeSurface(V3 Pc, V3 Nin, V3 base, double u, double v, double occl,
                       V3 Tn, V3 Bn, bool hasTB, int mat, uint64_t seed){
    if(mat==2) return shadeLeaf(Pc,Nin,base,u,v,Tn,Bn,hasTB,seed);
    V3 toCam = norm(Pc*-1.0);
    V3 N = norm(Nin);
    bool back = dot(N,toCam) < 0;                       // we're looking at the back of the petal
    if(back) N = N*-1.0;                                // two-sided: face the camera

    if(mat==0){                                        // ---------- rose petal ----------
        // pick front gradient or reverse colour for the visible face
        V3 albedo = (PC.reverseEnable && back) ? PC.reverseColor : base;
        if(hasTB){                                     // tangent-space "bump": fine veins + micro-relief
            double fade = std::sin(PI*clampd(u,0,1));               // veins fade out at base & tip
            double k    = 2.0*PI*PL.veinCount;
            double dh_dv = PL.veinDepth*fade*std::cos(k*v)
                         + PL.micro*std::sin(u*37.0+v*53.0);        // + micro chatter
            double dh_du = 0.04*std::cos(PI*u)*std::sin(k*v);       // slight cross-grain
            N = norm(N - (Tn*dh_du + Bn*dh_dv));
        }
        if(PL.veinTint>0){                             // the vein grooves read slightly darker
            double fade = std::sin(PI*clampd(u,0,1));
            albedo = albedo*(1.0 - PL.veinTint*fade*(0.5-0.5*std::sin(2.0*PI*PL.veinCount*v)));
        }
        double mott = 1.0 + PL.mottle*std::sin(u*6.0+v*2.0*PI+PL.mottlePhase)
                          + PL.mottle2*std::sin(u*23.0+PL.mottlePhase*1.3);   // soft mottling
        const V3 alb  = toLinear(albedo*mott);
        const V3 deep = deepen(alb);
        const double ao0 = std::max(0.0, 1.0 - PL.occlusion*(1.0-occl));                // baked occlusion,
        const double ao  = clampd(ao0*std::sqrt(std::sqrt(ao0)), 0.20, 1.0);              // deepened (^1.25)

        // key: soft, waxy falloff whose shadow side turns deep and saturated
        double dK = wrapDiffuse(N,g_Lkey,0.35);
        V3 key  = mul(mix(deep,alb,dK)*(dK*(0.4+0.6*ao)), LIGHT.keyCol);
        // fill & sky: cool and soft, they only reach what isn't tucked away
        V3 fill = mul(mix(deep,alb,0.5)*(softDiffuse(N,g_Lfill)*ao), LIGHT.fillCol)
                + mul(mix(deep,alb,0.3)*ao, LIGHT.ambient);
        // pale petals scatter light inside: their shadows stay luminous
        V3 scat = mul(alb*(0.5*PC.scatter*(1.0-dK)*(0.5+0.5*ao)), LIGHT.keyCol);
        // back light shining through the thin petal (thinnest at the rim), richest
        // when we look towards it
        double thin = 0.35 + 0.65*smoothstep(0.3,1.0,u);
        double thru = clampd(-dot(N,g_Lback),0,1);
        double fwd  = clampd(-dot(toCam,g_Lback),0,1); fwd=fwd*fwd*fwd;
        V3 trans = mul(mul(mix(alb,deep,0.4),V3(1.0,0.80,0.70))*(2.5*PL.backlight*thin*thru*(0.3+0.7*fwd)*ao),
                       LIGHT.backCol);
        // velvet: grazing angles catch the back light (and a little of the key)
        double f1   = 1.0-clampd(dot(N,toCam),0,1);
        double fres = f1*f1*f1*std::sqrt(f1);                                          // ^3.5
        V3 rimC = mix(alb, V3(1.0,0.92,0.86), 0.35);
        V3 rim  = mul(rimC*(fres*PL.rim*(0.25+0.75*softDiffuse(N,g_Lback))*(0.3+0.7*ao)), LIGHT.backCol)
                + mul(rimC*(fres*PL.rim*0.25*ao), LIGHT.keyCol);
        // broad satin sheen from the key
        double sheen = std::pow(clampd(dot(N,norm(g_Lkey+toCam)),0,1), 18.0)*PL.sheen*ao;
        return key + fill + scat + trans + rim + mul(V3(1.0,0.86,0.86)*sheen, LIGHT.keyCol);
    } else {                                           // ---------- stem / stalk / sepal ----------
        const V3 alb=toLinear(base);
        double dK=wrapDiffuse(N,g_Lkey,0.35);
        double spec = std::pow(clampd(dot(N,norm(g_Lkey+toCam)),0,1), 26.0)*0.12;
        double fres = std::pow(1.0-clampd(dot(N,toCam),0,1), 3.0);
        return mul(mix(deepen(alb),alb,dK)*dK, LIGHT.keyCol)
             + mul(alb*softDiffuse(N,g_Lfill), LIGHT.fillCol)
             + mul(alb, LIGHT.ambient)
             + mul(alb*(fres*0.35*(0.3+0.7*softDiffuse(N,g_Lback))), LIGHT.backCol)
             + mul(V3(1.0,1.0,0.9)*spec, LIGHT.keyCol);
    }
}
static inline V3 toneMap(V3 c){ return toDisplay(V3(aces(c.x), aces(c.y), aces(c.z))); }

static bool g_dew=false, g_focus=false, g_vignette=false, g_backdrop=false, g_breeze=false;
static double g_time=0.0;                  // seconds: drives the breeze and the sparkles' twinkling

struct DewLayer { double cell, density; uint64_t salt; };   // grid spacing, chance per cell
static const DewLayer DEW_LAYERS[2] = { {0.13, 0.50, 0x51ED27ull}, {0.05, 0.40, 0xD20F11ull} };
// Is this surface point under a drop? Drops are small spheres scattered on a
// jittered 3D grid; where one cuts the surface it leaves a round drop, larger
// or smaller depending on how deep the sphere sits. They gather best on
// surfaces that face up or down; steep ones hold fewer.
//   q: 0 at the drop's centre -> 1 at its rim;  tdir: centre -> this point, along the surface
static bool dewDropIn(const DewLayer& L, V3 Pw, V3 Nw, double& q, V3& tdir){
    const double DEW_CELL=L.cell;
    const double dens = L.density*(0.35 + 0.65*smoothstep(0.05,0.40,std::fabs(Nw.y)));
    const V3 g=Pw*(1.0/DEW_CELL);
    const int bx=(int)std::floor(g.x-0.5), by=(int)std::floor(g.y-0.5), bz=(int)std::floor(g.z-0.5);
    for(int dz=0;dz<2;++dz) for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx){
        const int cx=bx+dx, cy=by+dy, cz=bz+dz;
        const uint64_t h=hash64(g_seed*L.salt ^ hash64(((uint64_t)(cx&0x1FFFFF)<<42)
                                ^ ((uint64_t)(cy&0x1FFFFF)<<21) ^ (uint64_t)(cz&0x1FFFFF)));
        if(hashUnit(h,0) >= dens) continue;
        const V3 C=V3(cx+0.25+0.5*hashUnit(h,1), cy+0.25+0.5*hashUnit(h,2), cz+0.25+0.5*hashUnit(h,3))*DEW_CELL;
        const double r=DEW_CELL*(0.10+0.15*hashUnit(hash64(h),0));
        const V3 d=Pw-C; const double hn=dot(d,Nw), rc2=r*r-hn*hn;
        if(rc2<=0) continue;
        const V3 dt=d-Nw*hn; const double dl=len(dt), rc=std::sqrt(rc2);
        if(dl>=rc) continue;
        q=dl/rc; tdir= dl>1e-9 ? dt*(1.0/dl) : V3(0,0,0);
        return true;
    }
    return false;
}
static bool dewDrop(V3 Pw, V3 Nw, double& q, V3& tdir){
    for(const DewLayer& L : DEW_LAYERS) if(dewDropIn(L,Pw,Nw,q,tdir)) return true;
    return false;
}
// a drop over a surface whose (linear, exposed) colour is `lin`; view space
static V3 shadeDew(V3 lin, V3 Nv, V3 toCam, V3 tdirV, double q){
    const double nz=std::sqrt(std::max(0.0,1.0-q*q));
    const V3 Nd=norm(tdirV*q + Nv*(0.15+0.85*nz));                  // the drop's domed surface
    V3 c=lin*(0.50+0.30*nz);                                         // the surface, seen through water
    V3 Lt=g_Lkey - Nv*dot(g_Lkey,Nv); const double lt=len(Lt);       // light focused through the drop
    if(lt>1e-6){                                                     // lands on its far side
        double cc=dot(tdirV, Lt*(-1.0/lt));
        c = c + lin*(1.8*smoothstep(0.35,1.0,cc)*smoothstep(0.30,0.65,q)*(1.0-smoothstep(0.72,0.92,q)));
    }
    c = c*(1.0 - 0.7*smoothstep(0.80,1.0,q));                        // dark meniscus at the rim
    const double s1=std::pow(clampd(dot(Nd,norm(g_Lkey +toCam)),0,1),220.0)*9.0;  // sharp key highlight
    const double s2=std::pow(clampd(dot(Nd,norm(g_Lback+toCam)),0,1), 60.0)*1.5;  // glint of the back light
    const double f1=1.0-clampd(dot(Nd,toCam),0,1);
    return c + (LIGHT.keyCol*s1 + LIGHT.backCol*s2 + V3(0.30,0.36,0.46)*(0.6*f1*f1*f1))*LIGHT.exposure;
}

// the plant bends from the foot of its stem, more the higher up; leaflets
// flutter and the petals tremble a little, each with its own phase
static V3 swayed(const Vert& v, int mat){
    const double t=g_time;
    const double h=clampd((v.p.y-g_stemYBot)/(g_bloomTopY-g_stemYBot),0.0,1.2), bend=h*h;
    V3 d(0.035*bend*(std::sin(0.9*t) + 0.35*std::sin(2.3*t+1.1)), 0.0,
         0.025*bend*std::sin(0.7*t+0.6));
    if(mat==2) d.y += 0.012*v.u*std::sin(3.1*t + v.p.x*6.0 + v.p.z*5.0);
    if(mat==0) d.y += 0.004*v.u*std::sin(2.3*t + v.p.x*9.0 + v.p.z*7.0);
    return v.p + d;
}

enum WireMode { WIRE_NONE=0, WIRE_LEFT, WIRE_RIGHT, WIRE_ALL, WIRE_MODES };
static int       g_wireMode  = WIRE_NONE;
static int       g_wireStep  = 3;         // draw every Nth grid line (1 = finest)
static const int WIRE_STEP_MAX = 16;
static double    g_wireWidth = 0.2;       // line half-width, in output pixels
static const V3  WIRE_COLOR{0.55, 1.00, 0.10};   // lime green
static bool         g_glow    = false;
static bool         g_sparkle = false;
static const double GLOW_RADIUS   = 3.0;  // box radius per blur pass, output pixels
static const double GLOW_STRENGTH = 1.25;
static const double SPARK_DENSITY = 0.22; // chance that a grid cell's line segment has a glint
static const double SPARK_SIZE    = 18.0; // ray length of a fully lit glint, output pixels

struct Spark { int x, y; float depth; double tw, score; uint64_t key; };
static std::vector<Spark> g_sparks;       // glint centres found during rasterization

static inline V3 bgColor(int y){             // dim vertical gradient backdrop
    double t=(double)y/RH;
    V3 top{0.07,0.09,0.11}, bot{0.02,0.03,0.04};
    return mix(top,bot,t);
}

static inline bool isWirePixel(int x){
    switch(g_wireMode){
        case WIRE_LEFT:  return x <  RW/2;
        case WIRE_RIGHT: return x >= RW/2;
        case WIRE_ALL:   return true;
        default:         return false;
    }
}

static inline V3 unpack(uint32_t p){ return {((p>>16)&0xFF)/255.0, ((p>>8)&0xFF)/255.0, (p&0xFF)/255.0}; }

static std::vector<uint32_t> g_bgPlain, g_bgGarden;
static inline const std::vector<uint32_t>& backdrop(){
    return (g_backdrop && !g_bgGarden.empty()) ? g_bgGarden : g_bgPlain;
}
static inline V3 bgAt(size_t o){ return unpack(backdrop()[o]); }

static void makePlainBackground(){
    g_bgPlain.resize((size_t)RW*RH);
    for(int y=0;y<RH;++y){
        uint32_t c=pack(bgColor(y));
        std::fill(g_bgPlain.begin()+(size_t)y*RW, g_bgPlain.begin()+(size_t)(y+1)*RW, c);
    }
}

// a dusky garden far behind the rose, as a camera focused on the bloom sees
// it: soft masses of foliage, a few distant roses of the same colour, and
// bokeh — round out-of-focus highlights. Kept dark so the rose stands out.
static void makeGardenBackground(){
    std::mt19937_64 r(hash64(g_seed ^ 0xB0CEull));             // its own stream: the plant is unaffected
    auto U=[&](double a,double b){ return std::uniform_real_distribution<double>(a,b)(r); };
    const int GW=RW/8+2, GH=RH/8+2;                            // painted coarse, then smoothly upsampled
    const double aspect=(double)RW/RH;
    std::vector<V3> lo((size_t)GW*GH);
    for(int gy=0;gy<GH;++gy) for(int gx=0;gx<GW;++gx)
        lo[(size_t)gy*GW+gx]=mix(V3(0.035,0.060,0.050), V3(0.050,0.055,0.035), (double)gy/(GH-1));
    auto blob=[&](double bx,double by,double br,V3 bc,double strength){
        for(int gy=0;gy<GH;++gy) for(int gx=0;gx<GW;++gx){
            double dx=((double)gx/(GW-1)-bx)*aspect, dy=(double)gy/(GH-1)-by;
            double w=std::exp(-(dx*dx+dy*dy)/(br*br))*strength;
            V3& c=lo[(size_t)gy*GW+gx]; c=mix(c,bc,w);
        }
    };
    for(int k=0;k<22;++k)                                      // foliage masses
        blob(U(-0.1,1.1),U(-0.1,1.1),U(0.06,0.22), V3(U(0.03,0.08),U(0.08,0.17),U(0.03,0.07)), 0.8);
    const V3 far=PC.stops.back().rgb*0.45;                      // distant roses
    for(int k=0,n=(int)U(3.0,7.0);k<n;++k)
        blob(U(0.0,1.0),U(0.1,0.8),U(0.025,0.05), far*U(0.6,1.1), 0.7);
    std::vector<V3> img((size_t)RW*RH);
    for(int y=0;y<RH;++y) for(int x=0;x<RW;++x){                // bilinear upsample
        double fx=(double)x/(RW-1)*(GW-1), fy=(double)y/(RH-1)*(GH-1);
        int x0=std::min((int)fx,GW-2), y0=std::min((int)fy,GH-2); double tx=fx-x0, ty=fy-y0;
        V3 a=mix(lo[(size_t)y0*GW+x0],     lo[(size_t)y0*GW+x0+1],     tx);
        V3 b=mix(lo[(size_t)(y0+1)*GW+x0], lo[(size_t)(y0+1)*GW+x0+1], tx);
        img[(size_t)y*RW+x]=mix(a,b,ty);
    }
    for(int k=0;k<28;++k){                                     // bokeh: soft discs, brighter rims
        const double cx=U(0,RW), cy=U(0,RH*0.8), rad=U(4.0,16.0)*SS, a=U(0.04,0.12);
        const V3 col = U(0,1)<0.3 ? far*1.6 : V3(U(0.35,0.6),U(0.55,0.8),U(0.25,0.4));
        for(int y=std::max(0,(int)(cy-rad));y<=std::min(RH-1,(int)(cy+rad));++y)
            for(int x=std::max(0,(int)(cx-rad));x<=std::min(RW-1,(int)(cx+rad));++x){
                double d=std::hypot(x-cx,y-cy)/rad;
                if(d>=1.0) continue;
                double m=smoothstep(1.0,0.85,d)*(0.7+0.3*smoothstep(0.5,0.95,d));
                img[(size_t)y*RW+x]=img[(size_t)y*RW+x] + col*(a*m);
            }
    }
    g_bgGarden.resize((size_t)RW*RH);
    for(size_t i=0;i<img.size();++i) g_bgGarden[i]=pack(img[i]);
}

static void clearFB(){
    const std::vector<uint32_t>& bg=backdrop();
    std::copy(bg.begin(), bg.end(), g_fb.begin());
    std::fill(g_zb.begin(), g_zb.end(), 1e30f);
    std::fill(g_wm.begin(),g_wm.end(),0.0f);
    g_sparks.clear();
}

static inline void addPix(size_t o, V3 c){  // additive blend into the render buffer
    uint32_t p=g_fb[o];
    V3 b{((p>>16)&0xFF)/255.0, ((p>>8)&0xFF)/255.0, (p&0xFF)/255.0};
    g_fb[o]=pack(b+c);
}

// one horizontal box-blur pass of radius r (src -> dst), running sum
static void boxBlurH(const std::vector<float>& src, std::vector<float>& dst, int r){
    const float inv=1.0f/(2*r+1);
    for(int y=0;y<RH;++y){
        const float* s=&src[(size_t)y*RW]; float* d=&dst[(size_t)y*RW];
        float acc=0;
        for(int x=-r;x<=r;++x) acc+=s[std::clamp(x,0,RW-1)];
        for(int x=0;x<RW;++x){
            d[x]=acc*inv;
            acc+=s[std::min(x+r+1,RW-1)] - s[std::max(x-r,0)];
        }
    }
}
static void boxBlurV(const std::vector<float>& src, std::vector<float>& dst, int r){
    const float inv=1.0f/(2*r+1);
    for(int x=0;x<RW;++x){
        float acc=0;
        for(int y=-r;y<=r;++y) acc+=src[(size_t)std::clamp(y,0,RH-1)*RW+x];
        for(int y=0;y<RH;++y){
            dst[(size_t)y*RW+x]=acc*inv;
            acc+=src[(size_t)std::min(y+r+1,RH-1)*RW+x] - src[(size_t)std::max(y-r,0)*RW+x];
        }
    }
}

static void applyGlow(){
    int r=std::max(1,(int)std::lround(GLOW_RADIUS*SS));
    for(int pass=0;pass<3;++pass){ boxBlurH(g_wm,g_tmp,r); boxBlurV(g_tmp,g_wm,r); }
    for(int y=0;y<RH;++y) for(int x=0;x<RW;++x){
        if(!isWirePixel(x)) continue;
        size_t o=(size_t)y*RW+x;
        if(g_wm[o]>1e-3f) addPix(o, WIRE_COLOR*(GLOW_STRENGTH*g_wm[o]));
    }
}

// a four-pointed star (long horizontal/vertical rays, short diagonal ones)
static void drawGlint(int cx,int cy,double I){
    const double R=(0.35+0.65*I)*SPARK_SIZE;              // ray length, output pixels
    const int    r=(int)std::ceil(R*SS);
    const V3     tint=mix(WIRE_COLOR, V3(1,1,1), 0.6);
    for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx){
        int x=cx+dx, y=cy+dy;
        if(x<0||y<0||x>=RW||y>=RH||!isWirePixel(x)) continue;
        double u=(double)dx/SS, v=(double)dy/SS;
        double rr   = u*u+v*v;
        double core = std::exp(-rr/2.0);
        double halo = 0.25*std::exp(-rr/(0.04*R*R));
        double rayH = std::pow(std::max(0.0,1.0-std::fabs(u)/R),1.5)*std::exp(-v*v/0.6);
        double rayV = std::pow(std::max(0.0,1.0-std::fabs(v)/R),1.5)*std::exp(-u*u/0.6);
        double p=(u+v)*0.7071, q=(u-v)*0.7071, Rd=0.4*R;
        double rayD = std::pow(std::max(0.0,1.0-std::fabs(p)/Rd),1.5)*std::exp(-q*q/0.5)
                    + std::pow(std::max(0.0,1.0-std::fabs(q)/Rd),1.5)*std::exp(-p*p/0.5);
        double k=I*(1.2*core + halo + (rayH+rayV) + 0.5*rayD);
        if(k>2e-3) addPix((size_t)y*RW+x, tint*k);
    }
}

struct FocusLook {                  // all in bloom radii
    double plane    = 0.6;          // plane of focus: this far in front of the rose's axis
    double nearZone = 1.2;          // sharp from this far in front of the plane ...
    double farZone  = 0.7;          // ... to this far behind it
    double softness = 1.2;          // then fully soft this much further on
};
static FocusLook FOCUS;
static std::vector<float> g_dofC[3], g_dofB1[3], g_dofB2[3], g_coc;
static void applyFocus(double axisDepth){
    const size_t N=(size_t)RW*RH;
    for(int ch=0;ch<3;++ch){ g_dofC[ch].resize(N); g_dofB1[ch].resize(N); g_dofB2[ch].resize(N); }
    for(size_t i=0;i<N;++i){ V3 c=unpack(g_fb[i]); g_dofC[0][i]=c.x; g_dofC[1][i]=c.y; g_dofC[2][i]=c.z; }
    const int r1=std::max(1,3*SS), r2=std::max(2,7*SS);
    for(int ch=0;ch<3;++ch){
        boxBlurH(g_dofC[ch], g_tmp, r1);  boxBlurV(g_tmp, g_dofB1[ch], r1);
        boxBlurH(g_dofB1[ch], g_tmp, r1); boxBlurV(g_tmp, g_dofB1[ch], r1);
    }
    const double R=g_bloomR, plane=axisDepth - FOCUS.plane*R;
    const double nearZ=FOCUS.nearZone*R, farZ=FOCUS.farZone*R, soft=FOCUS.softness*R;
    g_coc.assign(N,0.0f);                                     // how blurred each pixel should be
    for(size_t o=0;o<N;++o){
        if(g_zb[o]>1e29f) continue;
        const double dz=g_zb[o]-plane;                            // + behind the plane, - in front
        g_coc[o]=(float)clampd(dz>0 ? (dz-farZ)/soft : (-dz-nearZ)/soft, 0.0, 1.0);
    }
    std::vector<float>& spread=g_dofB2[0];                    // (scratch until the blur below)
    boxBlurH(g_coc,g_tmp,r2); boxBlurV(g_tmp,spread,r2);
    for(size_t o=0;o<N;++o) if(g_zb[o]>1e29f) g_coc[o]=std::min(1.0f, 2.0f*spread[o]);   // backdrop at edges
    for(int ch=0;ch<3;++ch){
        boxBlurH(g_dofB1[ch], g_tmp, r2); boxBlurV(g_tmp, g_dofB2[ch], r2);
        boxBlurH(g_dofB2[ch], g_tmp, r2); boxBlurV(g_tmp, g_dofB2[ch], r2);
    }
    for(int y=0;y<RH;++y) for(int x=0;x<RW;++x){
        const size_t o=(size_t)y*RW+x;
        if(isWirePixel(x)) continue;
        const double coc=g_coc[o];
        if(coc<=0) continue;
        V3 sh(g_dofC[0][o],g_dofC[1][o],g_dofC[2][o]), b1(g_dofB1[0][o],g_dofB1[1][o],g_dofB1[2][o]),
           b2(g_dofB2[0][o],g_dofB2[1][o],g_dofB2[2][o]);
        g_fb[o]=pack(coc<0.5 ? mix(sh,b1,coc*2.0) : mix(b1,b2,coc*2.0-1.0));
    }
}

static void applySparkles(){
    // keep only glints whose centre pixel survived the depth test, one pixel per glint
    std::vector<Spark> vis;
    for(const Spark& s : g_sparks)
        if(g_zb[(size_t)s.y*RW+s.x]==s.depth) vis.push_back(s);
    std::sort(vis.begin(),vis.end(),[](const Spark&a,const Spark&b){
        return a.key!=b.key ? a.key<b.key : a.score<b.score; });
    for(size_t i=0;i<vis.size();++i)
        if(i==0 || vis[i].key!=vis[i-1].key) drawGlint(vis[i].x, vis[i].y, vis[i].tw);
}

static void renderFrame(double yaw,double pitch){
    clearFB();
    const V3 center = g_center;
    const double camDist = g_camDist;
    const double f=(RH*0.5)/std::tan(FOV*0.5);
    double cy=std::cos(yaw),sy=std::sin(yaw), cp=std::cos(pitch),sp=std::sin(pitch);

    auto toCam=[&](V3 p)->V3{
        p = p - center;
        V3 r1( cy*p.x + sy*p.z, p.y, -sy*p.x + cy*p.z );
        V3 r2( r1.x, cp*r1.y - sp*r1.z, sp*r1.y + cp*r1.z );
        return V3(r2.x, r2.y, r2.z - camDist);
    };
    auto rotN=[&](V3 n)->V3{
        V3 r1( cy*n.x + sy*n.z, n.y, -sy*n.x + cy*n.z );
        return V3( r1.x, cp*r1.y - sp*r1.z, sp*r1.y + cp*r1.z );
    };

    for(const Tri& T : g_tris){
        V3 pc[3];
        if(g_breeze){ pc[0]=toCam(swayed(T.a,T.mat)); pc[1]=toCam(swayed(T.b,T.mat)); pc[2]=toCam(swayed(T.c,T.mat)); }
        else        { pc[0]=toCam(T.a.p); pc[1]=toCam(T.b.p); pc[2]=toCam(T.c.p); }
        V3 nc[3]={rotN(T.a.n),rotN(T.b.n),rotN(T.c.n)};
        const V3 base[3]={T.a.col,T.b.col,T.c.col};
        const uint64_t leafSeed=(uint64_t)T.mesh ^ (RANDOMNESS>0 ? hash64(g_seed) : 0);
        const double uu[3]={T.a.u,T.b.u,T.c.u}, vv[3]={T.a.v,T.b.v,T.c.v};
        if(pc[0].z>-0.05||pc[1].z>-0.05||pc[2].z>-0.05) continue;   // near clip

        // view-space tangent frame from positions + UVs (for petal bump mapping)
        V3 Tn(0,0,0),Bn(0,0,0); bool hasTB=false;
        if(T.mat!=1){
            V3 e1=pc[1]-pc[0], e2=pc[2]-pc[0];
            double du1=uu[1]-uu[0],dv1=vv[1]-vv[0],du2=uu[2]-uu[0],dv2=vv[2]-vv[0];
            double det=du1*dv2-du2*dv1;
            if(std::fabs(det)>1e-7){
                double r=1.0/det;
                Tn=(e1*dv2 - e2*dv1)*r; Bn=(e2*du1 - e1*du2)*r;
                if(len(Tn)>1e-6 && len(Bn)>1e-6){ Tn=norm(Tn); Bn=norm(Bn); hasTB=true; }
            }
        }

        double sx[3],syy[3],dep[3];
        for(int k=0;k<3;++k){
            double inv=1.0/(-pc[k].z);
            sx[k]=RW*0.5 + f*pc[k].x*inv;
            syy[k]=RH*0.5 - f*pc[k].y*inv;
            dep[k]=-pc[k].z;
        }
        int minx=(int)std::floor(std::min({sx[0],sx[1],sx[2]}));
        int maxx=(int)std::ceil (std::max({sx[0],sx[1],sx[2]}));
        int miny=(int)std::floor(std::min({syy[0],syy[1],syy[2]}));
        int maxy=(int)std::ceil (std::max({syy[0],syy[1],syy[2]}));
        minx=std::max(0,minx); miny=std::max(0,miny);
        maxx=std::min(RW-1,maxx); maxy=std::min(RH-1,maxy);
        double area=(sx[1]-sx[0])*(syy[2]-syy[0])-(sx[2]-sx[0])*(syy[1]-syy[0]);
        if(std::fabs(area)<1e-9) continue;
        double invArea=1.0/area;
        // for wireframe: grid coordinates scaled so that grid lines fall on integers,
        // plus their (constant, per-triangle) screen-space gradients, so the distance
        // in pixels to the nearest line is |A - round(A)| / |grad A|.
        const double nA = std::max(1.0, std::round(T.na/g_wireStep));
        const double nB = std::max(1.0, std::round(T.nb/g_wireStep));
        const double gA[3]={T.a.ga*nA, T.b.ga*nA, T.c.ga*nA};
        const double gB[3]={T.a.gb*nB, T.b.gb*nB, T.c.gb*nB};
        auto gradLen=[&](const double* g){
            double dx=((g[1]-g[0])*(syy[2]-syy[0]) - (g[2]-g[0])*(syy[1]-syy[0]))*invArea;
            double dy=((g[2]-g[0])*(sx[1]-sx[0])  - (g[1]-g[0])*(sx[2]-sx[0]) )*invArea;
            return std::max(1e-9, std::hypot(dx,dy));
        };
        const double invGradA = 1.0/gradLen(gA), invGradB = 1.0/gradLen(gB);
        const double lineW = g_wireWidth*SS;
        for(int y=miny;y<=maxy;++y) for(int x=minx;x<=maxx;++x){
            double px=x+0.5, py=y+0.5;
            double w0=((sx[1]-px)*(syy[2]-py)-(sx[2]-px)*(syy[1]-py))*invArea;
            double w1=((sx[2]-px)*(syy[0]-py)-(sx[0]-px)*(syy[2]-py))*invArea;
            double w2=1.0-w0-w1;
            if(w0<0||w1<0||w2<0) continue;
            double d=w0*dep[0]+w1*dep[1]+w2*dep[2];
            size_t o=(size_t)y*RW+x;
            if(d>=g_zb[o]) continue;
            double u=w0*uu[0]+w1*uu[1]+w2*uu[2], v=w0*vv[0]+w1*vv[1]+w2*vv[2];
            if(T.mat==2 && std::fabs(v)>leafLimit(u,v)) continue;   // cut the leaflet's serrated margin
            g_zb[o]=(float)d;
            if(isWirePixel(x)){
                // hidden-line wireframe: the surface is painted with the backdrop
                // (and still writes depth, so it hides lines behind it); only a band
                // around each grid line is drawn in lime.
                double A=w0*gA[0]+w1*gA[1]+w2*gA[2], B=w0*gB[0]+w1*gB[1]+w2*gB[2];
                double eA=std::fabs(A-std::round(A))*invGradA;
                double eB=std::fabs(B-std::round(B))*invGradB;
                double e=std::min(eA,eB);
                if(T.mat==2)                                   // outline the serrated leaflet margin
                    e=std::min(e, (leafLimit(u,v)-std::fabs(v))*0.5*nB*invGradB);
                double a=clampd(1.0 - (e-lineW)/SS, 0.0, 1.0);    // solid core, 1px anti-aliased fringe
                g_fb[o]=pack(mix(bgAt(o), WIRE_COLOR, a));
                g_wm[o]=(float)a;
                if(g_sparkle && e<0.5*SS){
                    // glints sit at a hashed spot inside each grid cell's stretch of line,
                    // so they stay glued to the surface as the rose turns
                    bool   onA   = eA<=eB;
                    double along = onA ? B : A;
                    double cell  = std::floor(along);
                    uint64_t h = hash64(((uint64_t)T.mesh<<48) ^ ((uint64_t)onA<<47) ^ ((uint64_t)g_wireStep<<40)
                                        ^ ((uint64_t)(int64_t)std::round(onA?A:B)<<20) ^ (uint64_t)(int64_t)cell);
                    if(hashUnit(h,0) < SPARK_DENSITY){
                        double centre = cell + 0.2 + 0.6*hashUnit(h,1);
                        double dAlong = std::fabs(along-centre)*(onA ? invGradB : invGradA);
                        if(dAlong<0.75*SS){
                            double rate = 0.35 + 0.55*hashUnit(h,2);
                            double tw = std::pow(std::max(0.0, std::sin(2.0*PI*(g_time*rate + hashUnit(h,3)))), 12.0);
                            if(tw>0.02) g_sparks.push_back({x,y,(float)d,tw,dAlong+e,h});
                        }
                    }
                }
                continue;
            }
            // interpolate surface attributes, then shade per pixel
            V3 P{ w0*pc[0].x+w1*pc[1].x+w2*pc[2].x, w0*pc[0].y+w1*pc[1].y+w2*pc[2].y, w0*pc[0].z+w1*pc[1].z+w2*pc[2].z };
            V3 Nn{ w0*nc[0].x+w1*nc[1].x+w2*nc[2].x, w0*nc[0].y+w1*nc[1].y+w2*nc[2].y, w0*nc[0].z+w1*nc[1].z+w2*nc[2].z };
            V3 col{ w0*base[0].x+w1*base[1].x+w2*base[2].x, w0*base[0].y+w1*base[1].y+w2*base[2].y, w0*base[0].z+w1*base[1].z+w2*base[2].z };
            double occl=w0*T.a.ao+w1*T.b.ao+w2*T.c.ao;
            V3 lin=shadeSurface(P,Nn,col,u,v,occl,Tn,Bn,hasTB,T.mat,leafSeed)*LIGHT.exposure;
            if(g_dew && T.mat!=1){                          // drops sit on petals and leaves
                V3 Pw=T.a.p*w0 + T.b.p*w1 + T.c.p*w2, Nw=norm(T.a.n*w0 + T.b.n*w1 + T.c.n*w2);
                double q; V3 tdir;
                if(dewDrop(Pw,Nw,q,tdir)){
                    V3 toC=norm(P*-1.0), Nv=norm(Nn);
                    if(dot(Nv,toC)<0) Nv=Nv*-1.0;
                    lin=shadeDew(lin, Nv, toC, rotN(tdir), q);
                }
            }
            g_fb[o]=pack(toneMap(lin));
        }
    }
    if(g_wireMode!=WIRE_NONE){
        if(g_glow)    applyGlow();
        if(g_sparkle) applySparkles();
    }
    if(g_focus && g_wireMode!=WIRE_ALL) applyFocus(-toCam(g_focusPt).z);
    // resolve supersampling: box-downsample RW*RH -> WIN_W*WIN_H (and vignette, V)
    double inv=1.0/(SS*SS);
    for(int y=0;y<WIN_H;++y) for(int x=0;x<WIN_W;++x){
        double r=0,g=0,b=0;
        for(int sy=0;sy<SS;++sy) for(int sx=0;sx<SS;++sx){
            uint32_t c=g_fb[(size_t)(y*SS+sy)*RW+(x*SS+sx)];
            r+=(c>>16)&0xFF; g+=(c>>8)&0xFF; b+=c&0xFF;
        }
        double k=inv;
        if(g_vignette){
            double dx=(x+0.5-WIN_W*0.5)/(WIN_H*0.5), dy=(y+0.5-WIN_H*0.5)/(WIN_H*0.5);
            k*=1.0 - 0.55*smoothstep(0.35,1.30,std::sqrt(dx*dx+dy*dy));
        }
        g_out[(size_t)y*WIN_W+x]=(0xFFu<<24)|((uint32_t)(r*k+0.5)<<16)|((uint32_t)(g*k+0.5)<<8)|(uint32_t)(b*k+0.5);
    }
}

static void buildScene(){
    randomizeBloom();
    randomizePetalColors();
    randomizeLighting();
    buildRose();
    // find the lowest point of the bloom to attach the stem & sepals
    double minY=1e30; V3 lowest;
    for(const Tri&T:g_tris) for(const Vert*v:{&T.a,&T.b,&T.c})
        if(v->p.y<minY){ minY=v->p.y; lowest=v->p; }
    V3 attach(0.0, minY+0.02, 0.0);
    // a real bloom rarely stands perfectly upright: let it nod a little,
    // pivoting about the base of the petals (the origin)
    const double nod=RANDOMNESS*rnd(0.0,7.0)*PI/180.0, nodDir=rnd(0.0,2.0*PI);
    if(nod>1e-6){
        const V3 k(std::cos(nodDir),0,std::sin(nodDir));
        const double c=std::cos(nod), sn=std::sin(nod);
        auto rot=[&](V3 v){ return v*c + cross(k,v)*sn + k*(dot(k,v)*(1.0-c)); };  // Rodrigues
        for(Tri& T : g_tris) for(Vert* v : {&T.a,&T.b,&T.c}){ v->p=rot(v->p); v->n=rot(v->n); }
    }
    buildSepals(attach);
    randomizeStem();
    buildStem(attach.y, attach.y-3.2);
    buildThorns();
    // three or four compound leaves spiralling down the stem (~143 deg apart), lower ones larger
    const int    nLeaves = 3 + (rnd(0.0,1.0) < 0.45*RANDOMNESS ? 1 : 0);
    const double scales[4] = {1.15, 1.30, 1.40, 1.45};
    double t  = 0.14 + RANDOMNESS*rnd(-0.03,0.03);
    double az = 0.35 + RANDOMNESS*rnd(-PI,PI);
    for(int i=0;i<nLeaves;++i){
        int pairs = 2;                                    // 5 leaflets is the norm; the top leaf
        if(i==0 && rnd(0.0,1.0) < 0.35*RANDOMNESS) pairs=1;        // may have 3, low ones 7
        else if(i>=2 && rnd(0.0,1.0) < 0.35*RANDOMNESS) pairs=3;
        const double youth = i==0 ? 0.8 : i==1 ? 0.3 : 0.0;       // the top leaf is the newest
        buildRoseLeaf(t, az, scales[i]*vary(1.0,0.10), vary(0.45-0.05*i,0.25), pairs, youth);
        t  += 0.15 + RANDOMNESS*rnd(-0.03,0.03);
        az += 2.5  + RANDOMNESS*rnd(-0.4,0.4);
    }

    // auto-frame the camera on the bloom and the foliage, whatever the parameters.
    // The look-at point stays on the rose's vertical axis (x=z=0), so the
    // wireframe split still cuts the bloom in half.
    double top=-1e30, bot=1e30, rmax=0;
    for(const Tri&T:g_tris) if(T.mat!=1) for(const Vert*v:{&T.a,&T.b,&T.c}){
        if(T.mat==0) top=std::max(top,v->p.y+0.15);            // ...and over the bloom
        if(T.mat==2) bot=std::min(bot,v->p.y-0.12);            // small margin under the lowest leaf
        rmax=std::max(rmax,std::hypot(v->p.x,v->p.z));        // reach from the axis, at any yaw
    }
    double bloomTop=-1e30, bloomR=0;
    for(const Tri&T:g_tris) if(T.mat==0) for(const Vert*v:{&T.a,&T.b,&T.c}){
        bloomTop=std::max(bloomTop,v->p.y); bloomR=std::max(bloomR,std::hypot(v->p.x,v->p.z));
    }
    g_focusPt=V3(0.0, 0.5*(bloomTop+attach.y), 0.0); g_bloomR=bloomR; g_bloomTopY=bloomTop;
    double halfH=0.5*(top-bot), tanV=std::tan(FOV*0.5), tanH=tanV*WIN_W/WIN_H;
    g_center  = V3(0.0, 0.5*(top+bot), 0.0);
    g_camDist = std::max(halfH/tanV, rmax/tanH)*1.04;
}

// grow the whole plant from a seed (same seed -> same rose)
static void growRose(uint64_t seed){
    g_seed=seed; g_rng.seed(seed);
    g_tris.clear(); g_nextMeshId=0;
    buildScene();
    makeGardenBackground();
    fprintf(stderr,"seed: %llu  (%s, petals: %d, triangles: %zu)\n",
            (unsigned long long)seed, g_familyName, RP.numPetals, g_tris.size());
}

#ifdef HEADLESS
static void applyPreset(int p){
    switch(p){
        case 0: /* default deep red gradient — leave PC alone */ break;

        case 1: // Yellow throat -> red rim ("Peace"-style bicolour)
            PC.stops = {
                {0.0, V3{0.97, 0.82, 0.25}},
                {0.6, V3{0.96, 0.42, 0.18}},
                {1.0, V3{0.78, 0.06, 0.10}} };
            PC.axisRadial = 1.0; PC.axisDepth = 0.0; PC.bias = 0.8;
            break;

        case 2: // Cream petals with a crisp pink picotee edge
            PC.stops = {
                {0.0, V3{1.00, 0.96, 0.86}},
                {1.0, V3{0.98, 0.92, 0.82}} };
            PC.edgeEnable = true;
            PC.edgeColor  = V3{0.86, 0.18, 0.32};
            PC.edgeWidth  = 0.22;
            break;

        case 3: // Outer petals soft pink, inner petals deep magenta (depth gradient)
            PC.stops = {
                {0.0, V3{0.99, 0.78, 0.80}},
                {0.5, V3{0.92, 0.32, 0.46}},
                {1.0, V3{0.45, 0.04, 0.18}} };
            PC.axisRadial = 0.25; PC.axisDepth = 1.0; PC.bias = 1.0;
            break;

        case 4: // Bicolour reverse — red face, golden underside (a classic rose look)
            PC.stops = {
                {0.0, V3{0.55, 0.04, 0.10}},
                {1.0, V3{0.92, 0.16, 0.18}} };
            PC.reverseEnable = true;
            PC.reverseColor  = V3{0.96, 0.78, 0.30};
            break;
    }
}
int main(int argc,char**argv){
    double yaw  = argc>1? atof(argv[1])*PI/180.0 : 35.0*PI/180.0;
    double pitch= argc>2? atof(argv[2])*PI/180.0 : 14.0*PI/180.0;
    if(argc>3) SS=std::max(1,atoi(argv[3]));
    int preset = argc>4? atoi(argv[4]) : 0;
    if(argc>5) g_wireMode=std::max(0,atoi(argv[5]))%WIRE_MODES;
    if(argc>6) g_wireStep=std::clamp(atoi(argv[6]),1,WIRE_STEP_MAX);
    if(argc>7){ int fx=atoi(argv[7]); g_glow=fx&1; g_sparkle=fx&2; g_dew=fx&4; g_focus=fx&8;
                g_vignette=fx&16; g_backdrop=fx&32; g_breeze=fx&64; }
    if(argc>8) g_time=atof(argv[8]);
    uint64_t seed = argc>9 ? std::strtoull(argv[9],nullptr,10) : 1;
    applyPreset(preset);
    g_pickFamily = (preset==0);                      // a chosen preset keeps its colours
    setupBuffers();
    growRose(seed);
    fprintf(stderr,"render %dx%d, SS=%d, preset=%d\n", RW, RH, SS, preset);
    renderFrame(yaw,pitch);
    FILE*fp=fopen("rose3d.ppm","wb");
    fprintf(fp,"P6\n%d %d\n255\n",WIN_W,WIN_H);
    for(int i=0;i<WIN_W*WIN_H;++i){ uint32_t c=g_out[i];
        unsigned char rgb[3]={(unsigned char)(c>>16),(unsigned char)(c>>8),(unsigned char)c};
        fwrite(rgb,1,3,fp); }
    fclose(fp);
    fprintf(stderr,"wrote rose3d.ppm (%dx%d)\n",WIN_W,WIN_H);
    return 0;
}
#else
int main(int argc,char**argv){
    if(SDL_Init(SDL_INIT_VIDEO)!=0){ fprintf(stderr,"SDL_Init: %s\n",SDL_GetError()); return 1; }
    SDL_Window* win=SDL_CreateWindow("3D Rose",
        SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,WIN_W,WIN_H,SDL_WINDOW_SHOWN);
    SDL_Renderer* ren=SDL_CreateRenderer(win,-1,SDL_RENDERER_ACCELERATED|SDL_RENDERER_PRESENTVSYNC);
    SDL_Texture* tex=SDL_CreateTexture(ren,SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,WIN_W,WIN_H);

    setupBuffers();
    std::random_device rd;
    auto regrow=[&](uint64_t seed){
        growRose(seed);
        char title[160];
        snprintf(title,sizeof title,"3D Rose — seed %llu   (W P/L G S wireframe | D F V B M effects | R new rose)",
                 (unsigned long long)seed);
        SDL_SetWindowTitle(win,title);
    };
    regrow(argc>1 ? std::strtoull(argv[1],nullptr,10) : (uint64_t)rd());
    fprintf(stderr,"render %dx%d, SS=%d\n", RW, RH, SS);

    double yaw=0.4, pitch=14.0*PI/180.0; bool spin=true, run=true;
    Uint32 prev=SDL_GetTicks();
    while(run){
        SDL_Event e;
        while(SDL_PollEvent(&e)){
            if(e.type==SDL_QUIT) run=false;
            else if(e.type==SDL_KEYDOWN){
                switch(e.key.keysym.sym){
                    case SDLK_ESCAPE: case SDLK_q: run=false; break;
                    case SDLK_SPACE: spin=!spin; break;
                    case SDLK_w:     g_wireMode=(g_wireMode+1)%WIRE_MODES; break;
                    case SDLK_p:     g_wireStep=std::max(1,g_wireStep-1); break;             // finer
                    case SDLK_l:     g_wireStep=std::min(WIRE_STEP_MAX,g_wireStep+1); break; // chunkier
                    case SDLK_g:     g_glow=!g_glow; break;
                    case SDLK_s:     g_sparkle=!g_sparkle; break;
                    case SDLK_r:     regrow((uint64_t)rd()); break;
                    case SDLK_d:     g_dew=!g_dew; break;
                    case SDLK_f:     g_focus=!g_focus; break;
                    case SDLK_v:     g_vignette=!g_vignette; break;
                    case SDLK_b:     g_backdrop=!g_backdrop; break;
                    case SDLK_m:     g_breeze=!g_breeze; break;
                    case SDLK_LEFT:  yaw-=0.08; break;
                    case SDLK_RIGHT: yaw+=0.08; break;
                    case SDLK_UP:    pitch=clampd(pitch+0.05,-1.3,1.3); break;
                    case SDLK_DOWN:  pitch=clampd(pitch-0.05,-1.3,1.3); break;
                }
            }
        }
        Uint32 now=SDL_GetTicks(); double dt=(now-prev)/1000.0; prev=now;
        if(spin) yaw+=0.5*dt;
        g_time=now/1000.0;

        renderFrame(yaw,pitch);
        SDL_UpdateTexture(tex,nullptr,g_out.data(),WIN_W*sizeof(uint32_t));
        SDL_RenderClear(ren);
        SDL_RenderCopy(ren,tex,nullptr,nullptr);
        SDL_RenderPresent(ren);
    }
    SDL_DestroyTexture(tex); SDL_DestroyRenderer(ren); SDL_DestroyWindow(win); SDL_Quit();
    return 0;
}
#endif
