
header = """
<div class="fixed top-0 left-0 right-0 -z-40 h-[35vh] bg-cover bg-center bg-no-repeat" 
     style="background-image: url('https://images.unsplash.com/photo-1437846972679-9e6e537be46e?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=2071');
            -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 0%, rgba(0,0,0,0.95) 20%, rgba(0,0,0,0.8) 40%, rgba(0,0,0,0.6) 60%, rgba(0,0,0,0.3) 80%, rgba(0,0,0,0.1) 90%, rgba(0,0,0,0) 100%);
            mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 0%, rgba(0,0,0,0.95) 20%, rgba(0,0,0,0.8) 40%, rgba(0,0,0,0.6) 60%, rgba(0,0,0,0.3) 80%, rgba(0,0,0,0.1) 90%, rgba(0,0,0,0) 100%);">
</div>
<div class="flex flex-col items-center justify-center text-center p-3 my-2 md:my-4">
  <div class="mb-32">
  </div>
  <div id="user-greeting" class="text-3xl sm:text-4xl font-bold text-gray-900 mb-2">Welcome to Flight Agent</div>
  <p class="text-gray-600 text-base sm:text-lg mt-1 mb-3">
    Your AI-powered assistant for finding and booking flights with real-time prices
  </p>
  <div class="flex justify-center items-center gap-2 mt-2 mb-2 flex-wrap">
    <span class="px-2 py-1 bg-gray-50 text-gray-700 rounded-full text-xs">Real-time prices</span>
    <span class="px-2 py-1 bg-gray-50 text-gray-700 rounded-full text-xs">Live flight data</span>
    <span class="px-2 py-1 bg-gray-50 text-gray-700 rounded-full text-xs">AI-powered</span>
  </div>
  <div class="flex justify-center items-center mt-2">
    <a href="https://github.com/Cycls/Flight-Agent" target="_blank" rel="noopener noreferrer" class="flex items-center gap-2 px-3 py-1 bg-black text-white rounded-lg hover:bg-gray-800 transition-all no-underline">
      <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
      </svg>
      <span class="text-sm font-medium">GitHub</span>
    </a>
  </div>
</div>
"""

intro = """
---

<div class="py-1">
  <div class="flex flex-wrap gap-3 justify-center">
    <a href="https://cycls.com/send/${encodeURIComponent('Search flights from NYC to London tomorrow')}" class="group relative inline-flex items-center justify-center px-4 py-2 overflow-hidden font-medium text-gray-700 border-2 border-gray-300 rounded-xl shadow-lg bg-gradient-to-br from-gray-50 to-white focus:outline-none hover:border-gray-400 hover:shadow-xl transition-all whitespace-nowrap text-sm">
      <span>Search flights from NYC to London tomorrow</span>
    </a>
    <a href="https://cycls.com/send/${encodeURIComponent('Buscar vuelos de Madrid a Barcelona')}" class="group relative inline-flex items-center justify-center px-4 py-2 overflow-hidden font-medium text-gray-700 border-2 border-gray-300 rounded-xl shadow-lg bg-gradient-to-br from-gray-50 to-white focus:outline-none hover:border-gray-400 hover:shadow-xl transition-all whitespace-nowrap text-sm">
      <span>Buscar vuelos de Madrid a Barcelona</span>
    </a>
    <a href="https://cycls.com/send/${encodeURIComponent('Rechercher des vols Paris Rome')}" class="group relative inline-flex items-center justify-center px-4 py-2 overflow-hidden font-medium text-gray-700 border-2 border-gray-300 rounded-xl shadow-lg bg-gradient-to-br from-gray-50 to-white focus:outline-none hover:border-gray-400 hover:shadow-xl transition-all whitespace-nowrap text-sm">
      <span>Rechercher des vols Paris Rome</span>
    </a>
    <a href="https://cycls.com/send/${encodeURIComponent('東京からシンガポールへのフライトを検索')}" class="group relative inline-flex items-center justify-center px-4 py-2 overflow-hidden font-medium text-gray-700 border-2 border-gray-300 rounded-xl shadow-lg bg-gradient-to-br from-gray-50 to-white focus:outline-none hover:border-gray-400 hover:shadow-xl transition-all whitespace-nowrap text-sm">
      <span>東京からシンガポールへのフライトを検索</span>
    </a>
  </div>
</div>

"""
