# ==============================
# File: web/app.js
# ==============================
(() => {
  const W = { scene:null, camera:null, renderer:null, controls:null, clock:null, light:null, vrm:null };
  const state = { mouth:0, vowels:{aa:0,ee:0,ih:0,oh:0,ou:0}, blink:0, t:0, emotions:{happy:0,angry:0,sad:0,relaxed:0,surprised:0} };

  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);

  function init() {
    W.renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:false });
    W.renderer.setSize(window.innerWidth, window.innerHeight);
    W.renderer.setPixelRatio(window.devicePixelRatio);

    W.scene = new THREE.Scene();
    W.scene.background = new THREE.Color(0x0a0a0a);

    W.camera = new THREE.PerspectiveCamera(35, window.innerWidth/window.innerHeight, 0.1, 100);
    W.camera.position.set(0, 1.4, 2.2);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
    hemi.position.set(0, 1, 0);
    W.scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(1, 1.2, 1.5);
    W.scene.add(dir);

    const grid = new THREE.GridHelper(10, 10, 0x222222, 0x111111);
    grid.position.y = -1.0;
    W.scene.add(grid);

    W.clock = new THREE.Clock();
    window.addEventListener('resize', onResize);
    onResize();

    animate();
  }

  function onResize() {
    W.camera.aspect = window.innerWidth / window.innerHeight;
    W.camera.updateProjectionMatrix();
    W.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  function animate() {
    requestAnimationFrame(animate);
    const dt = W.clock.getDelta();
    state.t += dt;

    // Simple procedural blink
    const blinkSpeed = 6.0; // times/minute
    const phase = (state.t * blinkSpeed) % 1.0;
    let blink = 0.0;
    if (phase < 0.08) blink = 1.0 - (phase / 0.08); // close fast
    else if (phase < 0.16) blink = (phase - 0.08) / 0.08; // open fast
    state.blink = blink;

    // Idle breathing to move chest/head slightly
    if (W.vrm) {
      const s = 0.02 * Math.sin(state.t * 1.6);
      W.vrm.scene.position.y = s;
    }

    applyAvatarAnimation();
    W.renderer.render(W.scene, W.camera);
  }

  function applyAvatarAnimation() {
    if (!W.vrm) return;
    const em = W.vrm.expressionManager;
    if (em) {
      // Lip sync
      em.setValue('aa', state.vowels.aa ?? state.mouth);
      em.setValue('ee', state.vowels.ee ?? 0);
      em.setValue('ih', state.vowels.ih ?? 0);
      em.setValue('oh', state.vowels.oh ?? 0);
      em.setValue('ou', state.vowels.ou ?? 0);

      // Blink
      em.setValue('blinkL', state.blink);
      em.setValue('blinkR', state.blink);

      // Emotions (basic set)
      const E = state.emotions;
      em.setValue('happy', E.happy || 0);
      em.setValue('angry', E.angry || 0);
      em.setValue('sad', E.sad || 0);
      em.setValue('relaxed', E.relaxed || 0);
      em.setValue('surprised', E.surprised || 0);

      em.update();
    }
  }

  async function loadVRM(fileOrUrl) {
    const loader = new THREE.GLTFLoader();

    function onGLTF(gltf) {
      const vrm = THREE.VRM.from(gltf);
      if (W.vrm) {
        W.scene.remove(W.vrm.scene);
        W.vrm.dispose?.();
      }
      W.vrm = vrm;
      W.scene.add(vrm.scene);
      vrm.scene.rotation.y = Math.PI; // face camera
    }

    if (typeof fileOrUrl === 'string') {
      loader.load(fileOrUrl, (gltf) => {
        THREE.VRMUtils.removeUnnecessaryJoints(gltf.scene);
        THREE.VRMUtils.removeUnnecessaryVertices(gltf.scene);
        onGLTF(gltf);
      });
    } else {
      const file = fileOrUrl;
      const url = URL.createObjectURL(file);
      loader.load(url, (gltf) => {
        onGLTF(gltf);
        URL.revokeObjectURL(url);
      });
    }
  }

  const subtitleEl = document.getElementById('subtitle');
  const inputEl = document.getElementById('text');
  inputEl?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const text = inputEl.value.trim();
      if (text) {
        window.pywebview?.api.send_user_text(text);
        inputEl.value = '';
      }
    }
  });

  const vrmInput = document.getElementById('vrmfile');
  vrmInput?.addEventListener('change', (e) => {
    const f = e.target.files?.[0];
    if (f) loadVRM(f);
  });

  function tweenEmotion(name, target, durationMs) {
    const start = state.emotions[name] || 0;
    const delta = target - start;
    const T = Math.max(1, durationMs || 200);
    const t0 = performance.now();
    function step(now){
      const u = Math.min(1, (now - t0)/T);
      state.emotions[name] = start + delta * (u<0.5 ? 2*u*u : -1+(4-2*u)*u); // easeInOutQuad
      if (u < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // Expose a tiny JS API for Python to call
  window.avatar = {
    setSubtitle(text) { subtitleEl.textContent = text || ''; },
    updateMouth(payload) {
      state.mouth = payload.open ?? 0;
      state.vowels = payload;
    },
    setExpression(name, weight, durationMs) {
      tweenEmotion(name, weight, durationMs);
    },
    setEmotion(name, weight, durationMs) {
      // Alias for clarity
      tweenEmotion(name, weight, durationMs);
      // Optionally dampen others for exclusivity
      const others = ['happy','angry','sad','relaxed','surprised'].filter(n => n !== name);
      others.forEach(n => tweenEmotion(n, 0, durationMs));
    },
    loadVRM(path) { loadVRM(path); },
  };

  init();
})();
