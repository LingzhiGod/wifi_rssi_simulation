%% wifi_rssi_simulation.m
% High-fidelity WiFi RSSI simulation in a complex 3D indoor environment.
% This script demonstrates:
% 1) 3D room + material-aware obstacles
% 2) Multi-AP signal modeling with overlap/interference
% 3) LOS path loss + simplified multipath ray tracing (image method)
% 4) Shadowing + small-scale fading
% 5) Dynamic human attenuation
% 6) Structured outputs (.mat + .csv)
% 7) 3D visualization + RSSI heatmap + optional animation
%
% The script is self-contained and uses local functions only.

clear; clc; close all;
rng(42); % Reproducible fading/shadowing randomness

%% 1) Build simulation configuration and example scenario
cfg = defaultSimulationConfig();
[env, aps, humans, sim] = createExampleScenario(cfg);

% Create a 3D measurement grid (x,y,z points)
grid = createMeasurementGrid(env.roomDims, sim.gridStep, sim.gridZLevels);

%% 2) Run dynamic RSSI simulation
results = runRssiSimulation(env, aps, humans, grid, sim);

%% 3) Export outputs for localization experiments
exportSimulationData(results, grid, aps, sim);

%% 4) Visualizations
% 3D scene plot
figScene = figure('Name', '3D Indoor WiFi Environment', 'Color', 'w');
visualizeEnvironment3D(env, aps, humans, sim, results, 1);

% 2D RSSI heatmap slice for AP #1 at receiver height near 1.2 m at last time step
figHeat = figure('Name', 'RSSI Heatmap (AP1)', 'Color', 'w');
plotRssiSlice(grid, results, 1, results.nTimes, 1.2);

% 3D RSSI scatter for AP #1 at last time step
fig3D = figure('Name', '3D RSSI Scatter (AP1)', 'Color', 'w');
plotRssi3DScatter(grid, results, 1, results.nTimes);

% Optional animation for moving human + dynamic RSSI
if sim.enableAnimation
    animateDynamicRssi(env, aps, humans, grid, results, 1);
end

fprintf('\nSimulation complete.\n');
fprintf('MAT output : %s\n', results.outputMatFile);
fprintf('CSV output : %s\n', results.outputCsvFile);

%% ============================== Local Functions ==============================

function cfg = defaultSimulationConfig()
% Default parameters for propagation, fading, and simulation controls.

cfg = struct();

% Room configuration (meters)
cfg.roomDims = [10, 10, 3]; % [length, width, height]

% Grid configuration
cfg.gridStep = [0.5, 0.5, 0.5];
cfg.gridZLevels = 0.5:0.5:2.5;

% Time/dynamics
cfg.dt = 0.5;                 % seconds
cfg.simDuration = 20;         % seconds
cfg.enableRealtimeUpdate = true;

% Propagation model
cfg.pathLossExponent = 2.1;   % Adjustable: LOS/NLOS effective exponent
cfg.nlosPenaltyDb = 6;        % Additional NLOS attenuation when LOS is blocked
cfg.diffractionPenaltyDb = 4; % Simplified diffraction loss when blocked

% Multipath / ray tracing
cfg.enableRayTracing = true;
cfg.maxReflectionOrder = 2;   % Optional advanced feature: multiple reflections
cfg.maxReflectionComponents = 40; % Cap for runtime control

% Frequency selective fading (optional advanced feature)
cfg.enableFrequencySelective = true;
cfg.signalBandwidthHz = 20e6;
cfg.numSubcarriers = 12;

% Shadowing and small-scale fading
cfg.shadowSigmaDb = 3.0;      % Log-normal shadowing std (dB)
cfg.smallScaleModel = 'rician'; % 'rayleigh' | 'rician'
cfg.ricianK_dB = 6;           % Used when model is rician

% Interference/noise
cfg.ambientNoiseDbm = -95;    % Thermal + receiver noise floor
cfg.backgroundInterferenceDbm = -100;

% Output
cfg.outputDir = fullfile(pwd, 'output');

% Visualization
cfg.enableAnimation = true;
cfg.animationPause = 0.08;
end

function [env, aps, humans, sim] = createExampleScenario(cfg)
% Example scenario required by prompt:
% - 10m x 10m x 3m room
% - 3 APs
% - 2 obstacles
% - 1 moving human

% Environment
env = struct();
env.roomDims = cfg.roomDims;

% Room surface material properties for reflections (amplitude coefficients)
% Reflection gain uses (reflectionCoeff * (1-absorptionCoeff)).
env.surfaces = roomSurfacesWithMaterials(env.roomDims);

% Obstacles (axis-aligned boxes): [minXYZ], [maxXYZ], material, attenuation
% Obstacle 1: internal partition wall (concrete-like)
obs1.name = 'PartitionWall';
obs1.minXYZ = [4.5, 1.0, 0.0];
obs1.maxXYZ = [4.7, 8.5, 2.6];
obs1.material = struct('reflectionCoeff', 0.62, 'absorptionCoeff', 0.30, ...
                       'attenuationDbPerMeter', 14.0);

% Obstacle 2: furniture block (wooden cabinet)
obs2.name = 'Cabinet';
obs2.minXYZ = [7.0, 6.5, 0.0];
obs2.maxXYZ = [8.4, 8.6, 1.8];
obs2.material = struct('reflectionCoeff', 0.38, 'absorptionCoeff', 0.42, ...
                       'attenuationDbPerMeter', 5.5);

env.obstacles = [obs1, obs2];

% Access points (APs)
% Frequency chosen in 2.4 GHz band with partially overlapping channels.
aps(1) = makeAP('AP1', [1.2, 1.2, 2.7], 18, 2.412e9, 1, 20e6);
aps(2) = makeAP('AP2', [8.8, 1.5, 2.7], 17, 2.422e9, 3, 20e6); % overlaps AP1
aps(3) = makeAP('AP3', [5.2, 8.8, 2.7], 19, 2.437e9, 6, 20e6);

% Human model (moving attenuating object)
% Human is a vertical cylinder with trajectory in room.
humans(1).name = 'Person1';
humans(1).radius = 0.28;      % meters
humans(1).height = 1.75;      % meters
humans(1).attenuationDbPerMeter = 7.5; % through-body attenuation proxy
humans(1).baseXYZ = [1.0, 2.0, 0.0];
humans(1).waypoints = [
    1.0, 2.0, 0.0;
    3.0, 4.0, 0.0;
    6.5, 4.5, 0.0;
    8.5, 7.5, 0.0;
    3.0, 8.2, 0.0;
    1.0, 2.0, 0.0
];
humans(1).waypointTimes = linspace(0, cfg.simDuration, size(humans(1).waypoints, 1));
humans(1).loop = true;

% Simulation options
sim = struct();
sim.gridStep = cfg.gridStep;
sim.gridZLevels = cfg.gridZLevels;
sim.dt = cfg.dt;
sim.simDuration = cfg.simDuration;
sim.tVec = 0:cfg.dt:cfg.simDuration;
sim.nTimes = numel(sim.tVec);

sim.pathLossExponent = cfg.pathLossExponent;
sim.nlosPenaltyDb = cfg.nlosPenaltyDb;
sim.diffractionPenaltyDb = cfg.diffractionPenaltyDb;

sim.enableRayTracing = cfg.enableRayTracing;
sim.maxReflectionOrder = cfg.maxReflectionOrder;
sim.maxReflectionComponents = cfg.maxReflectionComponents;

sim.enableFrequencySelective = cfg.enableFrequencySelective;
sim.signalBandwidthHz = cfg.signalBandwidthHz;
sim.numSubcarriers = cfg.numSubcarriers;

sim.shadowSigmaDb = cfg.shadowSigmaDb;
sim.smallScaleModel = cfg.smallScaleModel;
sim.ricianK_dB = cfg.ricianK_dB;

sim.ambientNoiseDbm = cfg.ambientNoiseDbm;
sim.backgroundInterferenceDbm = cfg.backgroundInterferenceDbm;

sim.outputDir = cfg.outputDir;
sim.enableAnimation = cfg.enableAnimation;
sim.animationPause = cfg.animationPause;
sim.enableRealtimeUpdate = cfg.enableRealtimeUpdate;
end

function ap = makeAP(name, posXYZ, txPowerDbm, frequencyHz, channel, bandwidthHz)
ap = struct();
ap.name = name;
ap.pos = posXYZ(:).';
ap.txPowerDbm = txPowerDbm;
ap.frequencyHz = frequencyHz;
ap.channel = channel;
ap.bandwidthHz = bandwidthHz;
end

function surfaces = roomSurfacesWithMaterials(roomDims)
% Define six room boundary planes and material properties.
L = roomDims(1); W = roomDims(2); H = roomDims(3);

% reflectionCoeff: amplitude reflection coefficient [0,1]
% absorptionCoeff: used to reduce reflected amplitude
wallMat = struct('reflectionCoeff', 0.70, 'absorptionCoeff', 0.22);
floorMat = struct('reflectionCoeff', 0.55, 'absorptionCoeff', 0.30);
ceilMat = struct('reflectionCoeff', 0.60, 'absorptionCoeff', 0.25);

surfaces = struct('name', {}, 'normal', {}, 'value', {}, 'material', {});

surfaces(end+1) = makeSurface('x0', [1,0,0], 0, wallMat);
surfaces(end+1) = makeSurface('xL', [1,0,0], L, wallMat);
surfaces(end+1) = makeSurface('y0', [0,1,0], 0, wallMat);
surfaces(end+1) = makeSurface('yW', [0,1,0], W, wallMat);
surfaces(end+1) = makeSurface('z0', [0,0,1], 0, floorMat);
surfaces(end+1) = makeSurface('zH', [0,0,1], H, ceilMat);
end

function s = makeSurface(name, normal, value, mat)
s = struct();
s.name = name;
s.normal = normal(:).';
s.value = value;
s.material = mat;
end

function grid = createMeasurementGrid(roomDims, gridStep, zLevels)
% Create 3D measurement points where RSSI is evaluated.
xVec = 0:gridStep(1):roomDims(1);
yVec = 0:gridStep(2):roomDims(2);

% Keep z-levels inside room
zVec = zLevels(zLevels >= 0 & zLevels <= roomDims(3));

[X, Y, Z] = ndgrid(xVec, yVec, zVec);
points = [X(:), Y(:), Z(:)];

grid = struct();
grid.xVec = xVec;
grid.yVec = yVec;
grid.zVec = zVec;
grid.points = points;
grid.size3D = size(X);
grid.nPoints = size(points, 1);
end

function results = runRssiSimulation(env, aps, humans, grid, sim)
% Main simulation loop over time and grid points.
%
% Signal model per AP-receiver pair:
%   Received power (dBm) = Pt(dBm) - PathLoss(dB) - AdditionalLosses(dB)
%
% PathLoss base model:
%   FSPL(d0=1m) = 20*log10(4*pi*d0/lambda)
%   PL(d) = FSPL(1m) + 10*n*log10(d/1m)
%
% Additional effects:
% - Obstacle penetration attenuation from line-box intersection length
% - Human attenuation from line-cylinder intersection length
% - Simplified diffraction/NLOS penalty if LOS blocked
% - Multipath using image-source reflections (complex field summation)
% - Log-normal shadowing and Rayleigh/Rician small-scale fading
% - Co-channel interference with overlap factor + ambient noise

c = 299792458;
nAP = numel(aps);
nP = grid.nPoints;
nT = sim.nTimes;

signalDbm = nan(nP, nAP, nT);
interferenceDbm = nan(nP, nAP, nT);
sinrDb = nan(nP, nAP, nT);
effectiveRssiDbm = nan(nP, nAP, nT);

% Precompute image sources per AP for ray tracing
imageSources = cell(1, nAP);
for a = 1:nAP
    imageSources{a} = buildImageSources(aps(a), env, sim.maxReflectionOrder, sim.maxReflectionComponents);
end

% Precompute human positions per time
humanStateOverTime = cell(1, nT);
for ti = 1:nT
    humanStateOverTime{ti} = getHumanStateAtTime(humans, sim.tVec(ti));
end

for ti = 1:nT
    humanState = humanStateOverTime{ti};

    % Compute desired received signal power per AP for each grid point
    for p = 1:nP
        rx = grid.points(p, :);
        for a = 1:nAP
            signalDbm(p, a, ti) = computeReceivedPowerDbm(aps(a), rx, env, humanState, sim, imageSources{a}, c);
        end
    end

    % Interference/noise and SINR
    noiseLin = dbm2mW(sim.ambientNoiseDbm) + dbm2mW(sim.backgroundInterferenceDbm);

    for p = 1:nP
        sigLin = dbm2mW(signalDbm(p, :, ti));

        for a = 1:nAP
            interfLin = 0;
            for b = 1:nAP
                if b == a
                    continue;
                end
                overlap = channelOverlapFactor(aps(a), aps(b));
                interfLin = interfLin + overlap * sigLin(b);
            end

            totalIN = interfLin + noiseLin;
            snrLin = sigLin(a) / max(totalIN, realmin);

            interferenceDbm(p, a, ti) = mW2dbm(max(interfLin, realmin));
            sinrDb(p, a, ti) = 10 * log10(max(snrLin, realmin));

            % Effective RSSI proxy: desired signal discounted by interference pressure
            penaltyDb = 10 * log10(1 + interfLin / max(sigLin(a), realmin));
            effectiveRssiDbm(p, a, ti) = signalDbm(p, a, ti) - penaltyDb;
        end
    end
end

results = struct();
results.signalDbm = signalDbm;
results.interferenceDbm = interferenceDbm;
results.sinrDb = sinrDb;
results.effectiveRssiDbm = effectiveRssiDbm;
results.time = sim.tVec;
results.nTimes = nT;
results.humanStateOverTime = humanStateOverTime;
results.gridSize = grid.size3D;
results.createdAt = datetime('now');

results.outputMatFile = fullfile(sim.outputDir, 'wifi_rssi_results.mat');
results.outputCsvFile = fullfile(sim.outputDir, 'wifi_rssi_results.csv');
end

function prDbm = computeReceivedPowerDbm(ap, rx, env, humanState, sim, imageSources, c)
% Compute received power from one AP to one receiver point.
% Includes LOS + reflected rays + shadowing + small-scale fading.

lambda = c / ap.frequencyHz;
d0 = 1.0;
fspl1mDb = 20 * log10(4*pi*d0/lambda);

tx = ap.pos;
dLos = norm(rx - tx);
dLos = max(dLos, 0.5); % Avoid singular behavior at very short distances

% LOS path loss (log-distance)
plLosDb = fspl1mDb + 10 * sim.pathLossExponent * log10(dLos / d0);

% Penetration/obstruction losses along LOS
obsLossDb = computeObstacleLossDb(tx, rx, env.obstacles);
[humanLossDb, humanBlocked] = computeHumanLossDb(tx, rx, humanState);

% LOS blockage check for NLOS/diffraction penalty
obstacleBlocked = isBlockedByObstacles(tx, rx, env.obstacles);
isBlocked = obstacleBlocked || humanBlocked;

nlosDb = 0;
if isBlocked
    nlosDb = sim.nlosPenaltyDb + sim.diffractionPenaltyDb;
end

% Shadowing (log-normal in dB)
shadowDb = sim.shadowSigmaDb * randn();

% LOS component power
prLosDbm = ap.txPowerDbm - plLosDb - obsLossDb - humanLossDb - nlosDb + shadowDb;
prLosLin = dbm2mW(prLosDbm);

if ~sim.enableRayTracing
    prCombinedLin = prLosLin;
else
    % Add reflected components via image-source method
    [pathDistances, pathPowersLin] = reflectedComponents(ap, rx, env, imageSources, sim, fspl1mDb, d0);

    if isempty(pathDistances)
        prCombinedLin = prLosLin;
    else
        % Complex field summation: E_total = sum sqrt(P_k) * exp(-j*phase_k)
        % Optional frequency-selective averaging over multiple subcarriers.
        if sim.enableFrequencySelective
            fOffsets = linspace(-sim.signalBandwidthHz/2, sim.signalBandwidthHz/2, sim.numSubcarriers);
            pSub = zeros(1, numel(fOffsets));

            for k = 1:numel(fOffsets)
                fk = ap.frequencyHz + fOffsets(k);
                phaseLos = 2*pi*fk*dLos/c;
                E = sqrt(prLosLin) * exp(-1j * phaseLos);

                for m = 1:numel(pathDistances)
                    phaseRef = 2*pi*fk*pathDistances(m)/c;
                    E = E + sqrt(pathPowersLin(m)) * exp(-1j * phaseRef);
                end

                pSub(k) = abs(E).^2;
            end

            prCombinedLin = mean(pSub);
        else
            phaseLos = 2*pi*ap.frequencyHz*dLos/c;
            E = sqrt(prLosLin) * exp(-1j * phaseLos);

            for m = 1:numel(pathDistances)
                phaseRef = 2*pi*ap.frequencyHz*pathDistances(m)/c;
                E = E + sqrt(pathPowersLin(m)) * exp(-1j * phaseRef);
            end

            prCombinedLin = abs(E).^2;
        end
    end
end

% Small-scale fading gain
fadeLin = smallScaleFadingGain(sim.smallScaleModel, sim.ricianK_dB);
prFinalLin = prCombinedLin * fadeLin;

prDbm = mW2dbm(max(prFinalLin, realmin));
end

function [pathDistances, pathPowersLin] = reflectedComponents(ap, rx, env, imageSources, sim, fspl1mDb, d0)
% Compute reflected ray powers from prebuilt image sources.

pathDistances = [];
pathPowersLin = [];

for i = 1:numel(imageSources)
    src = imageSources(i);

    % Geometric path length from image source to receiver
    dRef = norm(rx - src.imagePos);
    dRef = max(dRef, 0.5);

    % Base path loss with same exponent
    plDb = fspl1mDb + 10 * sim.pathLossExponent * log10(dRef / d0);

    % Reflection loss from amplitude gain
    reflGain = max(src.totalReflectionGain, 1e-3);
    reflLossDb = -20 * log10(reflGain);

    % Additional attenuation due to crossing obstacles on reflected segment
    % (Approximate: line from image source to receiver)
    obsLossDb = computeObstacleLossDb(src.imagePos, rx, env.obstacles);

    % Reflections are usually non-LOS paths: add mild NLOS penalty scaled by order
    orderPenaltyDb = 1.5 * src.order;

    prRefDbm = ap.txPowerDbm - plDb - reflLossDb - obsLossDb - orderPenaltyDb;

    pathDistances(end+1) = dRef; %#ok<AGROW>
    pathPowersLin(end+1) = dbm2mW(prRefDbm); %#ok<AGROW>
end
end

function imageSources = buildImageSources(ap, env, maxOrder, maxComponents)
% Build image sources up to max reflection order using room boundary surfaces.
% Uses iterative image construction on room planes.

if maxOrder < 1
    imageSources = struct('imagePos', {}, 'totalReflectionGain', {}, 'order', {}, 'history', {});
    return;
end

% Order-0 source
seed.imagePos = ap.pos;
seed.totalReflectionGain = 1.0;
seed.order = 0;
seed.history = strings(1,0);

frontier = seed;
imageSources = repmat(seed, 1, 0);

for ord = 1:maxOrder
    nextFrontier = repmat(seed, 1, 0);
    idx = 0;

    for i = 1:numel(frontier)
        base = frontier(i);

        for s = 1:numel(env.surfaces)
            surf = env.surfaces(s);
            imgPos = mirrorPointAcrossPlane(base.imagePos, surf);

            reflAmp = surf.material.reflectionCoeff * (1 - surf.material.absorptionCoeff);
            newNode.imagePos = imgPos;
            newNode.totalReflectionGain = base.totalReflectionGain * max(reflAmp, 1e-3);
            newNode.order = ord;
            newNode.history = [base.history, string(surf.name)];

            % Skip immediate repeated surface for basic pruning
            if numel(newNode.history) >= 2 && newNode.history(end) == newNode.history(end-1)
                continue;
            end

            idx = idx + 1;
            nextFrontier(idx) = newNode;
        end
    end

    % Append this order
    if ~isempty(nextFrontier)
        imageSources = [imageSources, nextFrontier]; %#ok<AGROW>
    end

    frontier = nextFrontier;

    % Runtime cap
    if numel(imageSources) >= maxComponents
        imageSources = imageSources(1:maxComponents);
        break;
    end
end
end

function pMir = mirrorPointAcrossPlane(p, surface)
% Plane form: for axis-aligned room planes defined by normal and value.
n = surface.normal;
v = surface.value;

if abs(n(1)) > 0
    % x = v
    pMir = [2*v - p(1), p(2), p(3)];
elseif abs(n(2)) > 0
    % y = v
    pMir = [p(1), 2*v - p(2), p(3)];
else
    % z = v
    pMir = [p(1), p(2), 2*v - p(3)];
end
end

function lossDb = computeObstacleLossDb(p1, p2, obstacles)
% Sum attenuation from all obstacle crossings.
lossDb = 0;

for i = 1:numel(obstacles)
    obs = obstacles(i);
    lenInside = segmentBoxIntersectionLength(p1, p2, obs.minXYZ, obs.maxXYZ);
    if lenInside > 0
        % dB attenuation proportional to traversed length in material
        lossDb = lossDb + lenInside * obs.material.attenuationDbPerMeter;

        % Additional absorption term (small)
        lossDb = lossDb + 1.5 * obs.material.absorptionCoeff * lenInside;
    end
end
end

function blocked = isBlockedByObstacles(p1, p2, obstacles)
blocked = false;
for i = 1:numel(obstacles)
    lenInside = segmentBoxIntersectionLength(p1, p2, obstacles(i).minXYZ, obstacles(i).maxXYZ);
    if lenInside > 0.03
        blocked = true;
        return;
    end
end
end

function len = segmentBoxIntersectionLength(p1, p2, bmin, bmax)
% Slab method: compute segment length inside axis-aligned box.
d = p2 - p1;
tMin = 0;
tMax = 1;

for k = 1:3
    if abs(d(k)) < 1e-12
        if p1(k) < bmin(k) || p1(k) > bmax(k)
            len = 0;
            return;
        end
    else
        t1 = (bmin(k) - p1(k)) / d(k);
        t2 = (bmax(k) - p1(k)) / d(k);
        tNear = min(t1, t2);
        tFar = max(t1, t2);

        tMin = max(tMin, tNear);
        tMax = min(tMax, tFar);

        if tMin > tMax
            len = 0;
            return;
        end
    end
end

segLen = norm(d);
len = max(0, (tMax - tMin) * segLen);
end

function [lossDb, blocked] = computeHumanLossDb(p1, p2, humanState)
% Human represented as vertical cylinder.
lossDb = 0;
blocked = false;

for i = 1:numel(humanState)
    h = humanState(i);
    centerXY = h.baseXYZ(1:2);
    zMin = h.baseXYZ(3);
    zMax = h.baseXYZ(3) + h.height;

    lenInside = segmentCylinderIntersectionLength(p1, p2, centerXY, h.radius, zMin, zMax);
    if lenInside > 0
        blocked = true;
        lossDb = lossDb + lenInside * h.attenuationDbPerMeter;
    end
end
end

function len = segmentCylinderIntersectionLength(p1, p2, centerXY, radius, zMin, zMax)
% Length of segment inside finite vertical cylinder.

d = p2 - p1;
a = d(1)^2 + d(2)^2;

% XY interval for infinite cylinder
if a < 1e-12
    % Segment nearly vertical in XY: either always inside or always outside cylinder
    distXY = hypot(p1(1)-centerXY(1), p1(2)-centerXY(2));
    if distXY > radius
        len = 0;
        return;
    end
    tXY = [0, 1];
else
    b = 2 * ((p1(1)-centerXY(1))*d(1) + (p1(2)-centerXY(2))*d(2));
    c = (p1(1)-centerXY(1))^2 + (p1(2)-centerXY(2))^2 - radius^2;
    disc = b^2 - 4*a*c;

    if disc < 0
        len = 0;
        return;
    end

    sqrtDisc = sqrt(disc);
    t1 = (-b - sqrtDisc) / (2*a);
    t2 = (-b + sqrtDisc) / (2*a);
    tXY = sort([t1, t2]);
end

% Z interval
if abs(d(3)) < 1e-12
    if p1(3) < zMin || p1(3) > zMax
        len = 0;
        return;
    end
    tZ = [0, 1];
else
    tz1 = (zMin - p1(3)) / d(3);
    tz2 = (zMax - p1(3)) / d(3);
    tZ = sort([tz1, tz2]);
end

tStart = max([0, tXY(1), tZ(1)]);
tEnd = min([1, tXY(2), tZ(2)]);

if tEnd <= tStart
    len = 0;
else
    len = (tEnd - tStart) * norm(d);
end
end

function humansAtT = getHumanStateAtTime(humans, t)
% Interpolate each human position from waypoints.

humansAtT = humans;
for i = 1:numel(humans)
    h = humans(i);

    if h.loop
        t0 = h.waypointTimes(1);
        t1 = h.waypointTimes(end);
        T = t1 - t0;
        if T > 0
            tEff = mod(t - t0, T) + t0;
        else
            tEff = t;
        end
    else
        tEff = min(max(t, h.waypointTimes(1)), h.waypointTimes(end));
    end

    x = interp1(h.waypointTimes, h.waypoints(:,1), tEff, 'linear', 'extrap');
    y = interp1(h.waypointTimes, h.waypoints(:,2), tEff, 'linear', 'extrap');
    z = interp1(h.waypointTimes, h.waypoints(:,3), tEff, 'linear', 'extrap');

    humansAtT(i).baseXYZ = [x, y, z];
end
end

function overlap = channelOverlapFactor(apA, apB)
% Overlap factor in [0,1], approximated from center-frequency distance.
% 1 => same/fully overlapping channel, 0 => no overlap.

deltaF = abs(apA.frequencyHz - apB.frequencyHz);
sharedBW = 0.5 * (apA.bandwidthHz + apB.bandwidthHz);
overlap = max(0, 1 - deltaF / sharedBW);

% Clamp to [0,1]
overlap = min(max(overlap, 0), 1);
end

function g = smallScaleFadingGain(model, ricianK_dB)
% Return linear power gain for small-scale fading.

switch lower(model)
    case 'rayleigh'
        % Complex Gaussian h => |h|^2 is exponential (mean=1)
        h = (randn + 1j*randn) / sqrt(2);
        g = abs(h).^2;

    case 'rician'
        K = 10^(ricianK_dB / 10);
        % h = deterministic LOS + diffuse NLOS
        h = sqrt(K/(K+1)) + sqrt(1/(2*(K+1))) * (randn + 1j*randn);
        g = abs(h).^2;

    otherwise
        g = 1.0;
end
end

function exportSimulationData(results, grid, aps, sim)
% Export simulation outputs in MAT and CSV formats.

if ~exist(sim.outputDir, 'dir')
    mkdir(sim.outputDir);
end

% Save rich MATLAB structure
save(results.outputMatFile, 'results', 'grid', 'aps', 'sim', '-v7.3');

% Build flattened table for CSV
nP = size(results.signalDbm, 1);
nAP = size(results.signalDbm, 2);
nT = size(results.signalDbm, 3);

rows = nP * nAP * nT;
timeCol = zeros(rows, 1);
idxPtCol = zeros(rows, 1);
xCol = zeros(rows, 1);
yCol = zeros(rows, 1);
zCol = zeros(rows, 1);
apCol = strings(rows, 1);
sigCol = zeros(rows, 1);
intCol = zeros(rows, 1);
sinrCol = zeros(rows, 1);
effCol = zeros(rows, 1);

r = 1;
for ti = 1:nT
    for a = 1:nAP
        for p = 1:nP
            timeCol(r) = results.time(ti);
            idxPtCol(r) = p;
            xCol(r) = grid.points(p,1);
            yCol(r) = grid.points(p,2);
            zCol(r) = grid.points(p,3);
            apCol(r) = string(aps(a).name);
            sigCol(r) = results.signalDbm(p,a,ti);
            intCol(r) = results.interferenceDbm(p,a,ti);
            sinrCol(r) = results.sinrDb(p,a,ti);
            effCol(r) = results.effectiveRssiDbm(p,a,ti);
            r = r + 1;
        end
    end
end

T = table(timeCol, idxPtCol, xCol, yCol, zCol, apCol, sigCol, intCol, sinrCol, effCol, ...
    'VariableNames', {'time_s','point_id','x_m','y_m','z_m','ap_name', ...
                      'signal_dbm','interference_dbm','sinr_db','effective_rssi_dbm'});

writetable(T, results.outputCsvFile);
end

function visualizeEnvironment3D(env, aps, humans, ~, results, timeIdx)
% Plot room, obstacles, APs, and human(s) at a selected time.

hold on;
axis equal;
grid on;
view(3);

L = env.roomDims(1); W = env.roomDims(2); H = env.roomDims(3);

% Room wireframe
plotRoomWireframe(L, W, H);

% Obstacles
colors = [0.75 0.45 0.45; 0.45 0.55 0.75; 0.55 0.70 0.50; 0.75 0.70 0.40];
for i = 1:numel(env.obstacles)
    c = colors(mod(i-1,size(colors,1))+1, :);
    drawCuboid(env.obstacles(i).minXYZ, env.obstacles(i).maxXYZ, c, 0.4);
end

% AP markers
for a = 1:numel(aps)
    p = aps(a).pos;
    scatter3(p(1), p(2), p(3), 90, 'r', 'filled', 'MarkerEdgeColor', 'k');
    text(p(1)+0.1, p(2)+0.1, p(3)+0.05, aps(a).name, 'FontWeight', 'bold');
end

% Human trajectory and current position
for h = 1:numel(humans)
    wp = humans(h).waypoints;
    plot3(wp(:,1), wp(:,2), wp(:,3)+0.9, 'k--', 'LineWidth', 1.2);

    hs = results.humanStateOverTime{timeIdx}(h);
    drawCylinder(hs.baseXYZ, hs.radius, hs.height, [0.20 0.65 0.95], 0.55);
end

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('3D Environment at t = %.1f s', results.time(timeIdx)));
xlim([0 L]); ylim([0 W]); zlim([0 H]);
legend({'Room','Trajectory'}, 'Location', 'northeastoutside');
end

function plotRoomWireframe(L, W, H)
% Draw room edges.
V = [
    0 0 0;
    L 0 0;
    L W 0;
    0 W 0;
    0 0 H;
    L 0 H;
    L W H;
    0 W H
];
E = [
    1 2; 2 3; 3 4; 4 1;
    5 6; 6 7; 7 8; 8 5;
    1 5; 2 6; 3 7; 4 8
];
for i = 1:size(E,1)
    p1 = V(E(i,1),:); p2 = V(E(i,2),:);
    plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], 'k-', 'LineWidth', 1.0);
end
end

function drawCuboid(minXYZ, maxXYZ, faceColor, faceAlpha)
% Draw a solid axis-aligned box.

x = [minXYZ(1), maxXYZ(1)];
y = [minXYZ(2), maxXYZ(2)];
z = [minXYZ(3), maxXYZ(3)];

verts = [
    x(1) y(1) z(1);
    x(2) y(1) z(1);
    x(2) y(2) z(1);
    x(1) y(2) z(1);
    x(1) y(1) z(2);
    x(2) y(1) z(2);
    x(2) y(2) z(2);
    x(1) y(2) z(2)
];

faces = [
    1 2 3 4;
    5 6 7 8;
    1 2 6 5;
    2 3 7 6;
    3 4 8 7;
    4 1 5 8
];

patch('Vertices', verts, 'Faces', faces, ...
      'FaceColor', faceColor, 'FaceAlpha', faceAlpha, ...
      'EdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.7);
end

function drawCylinder(baseXYZ, radius, height, color, alphaVal)
% Draw a vertical cylinder for human body.

n = 40;
theta = linspace(0, 2*pi, n);
xc = baseXYZ(1) + radius * cos(theta);
yc = baseXYZ(2) + radius * sin(theta);

X = [xc; xc];
Y = [yc; yc];
Z = [baseXYZ(3) * ones(1,n); (baseXYZ(3)+height) * ones(1,n)];

surf(X, Y, Z, 'FaceColor', color, 'FaceAlpha', alphaVal, 'EdgeColor', 'none');
fill3(xc, yc, baseXYZ(3)+height*ones(1,n), color, 'FaceAlpha', alphaVal, 'EdgeColor', 'none');
end

function plotRssiSlice(gridData, results, apIdx, timeIdx, targetZ)
% 2D heatmap at z slice nearest to targetZ.

[~, zId] = min(abs(gridData.zVec - targetZ));
zVal = gridData.zVec(zId);

xVec = gridData.xVec;
yVec = gridData.yVec;

nx = numel(xVec);
ny = numel(yVec);

% Recover point indices for this z layer from ndgrid order
% ndgrid(x,y,z) => x fastest varying by first dimension in linear indexing.
% We reconstruct by filtering points with this z.
mask = abs(gridData.points(:,3) - zVal) < 1e-9;
pts = gridData.points(mask, :);
rssi = results.effectiveRssiDbm(mask, apIdx, timeIdx);

% Map to matrix
a = nan(nx, ny);
for i = 1:size(pts,1)
    ix = find(abs(xVec - pts(i,1)) < 1e-9, 1);
    iy = find(abs(yVec - pts(i,2)) < 1e-9, 1);
    a(ix, iy) = rssi(i);
end

imagesc(xVec, yVec, a.');
set(gca, 'YDir', 'normal');
axis equal tight;
colorbar;
colormap(turbo);
xlabel('X (m)'); ylabel('Y (m)');
title(sprintf('Effective RSSI (dBm), AP%d, t=%.1fs, z=%.2fm', ...
    apIdx, results.time(timeIdx), zVal));
end

function plotRssi3DScatter(gridData, results, apIdx, timeIdx)
% 3D scatter of RSSI across all grid points.
rssi = results.effectiveRssiDbm(:, apIdx, timeIdx);

scatter3(gridData.points(:,1), gridData.points(:,2), gridData.points(:,3), 22, rssi, 'filled');
axis equal;
grid on;
view(35, 26);
cb = colorbar;
cb.Label.String = 'Effective RSSI (dBm)';
colormap(turbo);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('3D RSSI Distribution (AP%d) at t=%.1f s', apIdx, results.time(timeIdx)));
end

function animateDynamicRssi(env, aps, humans, gridData, results, apIdx)
% Animate moving humans and RSSI heatmap over time.

fig = figure('Name', 'Dynamic Human + RSSI Animation', 'Color', 'w');

for ti = 1:results.nTimes
    clf(fig);

    subplot(1,2,1);
    hold on;
    visualizeEnvironment3D(env, aps, humans, struct(), results, ti);
    title(sprintf('Environment (t = %.1f s)', results.time(ti)));

    subplot(1,2,2);
    plotRssiSlice(gridData, results, apIdx, ti, 1.2);

    drawnow;
    pause(0.05);
end
end

function p = dbm2mW(dbm)
p = 10.^(dbm/10);
end

function dbm = mW2dbm(p)
dbm = 10*log10(p);
end
