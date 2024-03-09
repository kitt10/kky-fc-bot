(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[231],{4280:function(e,n,t){"use strict";t.d(n,{Z:function(){return g}});t(7294);var i=t(1664),r=t(917),o=t(950),c=t(5077),s=t(2668),l=t(5893),a=function(e,n){return(0,r.iv)({color:e||"inherit",fontSize:n+" !important"})},d=function(e){var n=e.color,t=e.fontSize,i=e.iconStyle,r=e.children;return(0,l.jsx)(s.Z,{css:[a(n,t),i],children:r})};t(2809);var u=t(1770),f=function(e){return(0,r.iv)({width:"100%",height:e.view.headerHeight,minHeight:e.view.headerHeight,zIndex:200,backgroundColor:e.colors.dark,color:e.colors.blond,display:"flex",flexDirection:"row",justifyContent:"center",alignItems:"center"})},h=function(e,n){return(0,r.iv)({height:"calc(".concat(e.view.headerHeight," - 20px)"),width:"auto",fontSize:"15px",color:n?e.colors.highlight:e.colors.blond,marginLeft:"25px",paddingLeft:"10px",paddingRight:"10px",display:"flex",flexDirection:"row",justifyContent:"center",alignItems:"center",cursor:"pointer",borderRadius:"5px",":hover":{backgroundColor:"#222222"}})},x=function(e){return(0,r.iv)({marginRight:"10px"})},g=function(e){var n,t=(0,o.H)(),s=t.style,a=t.page,g=(0,c.f)(),p=g.logged,v=g.sc,m=u.L.sc.useSc&&v.initialized;return(0,l.jsxs)("div",{css:f(s),children:[(0,l.jsx)("a",{href:"https://kky.zcu.cz",target:"blank",title:"Open the KKY website.",children:(0,l.jsx)("img",{css:h(s,!1),src:"/img/kky_logo.png"})}),(0,l.jsx)(i.default,{href:"/",children:(0,l.jsx)("div",{css:h(s,"index"==a),children:"Chatbot KKY ft. Fxck Cancer"})}),(0,l.jsx)(i.default,{href:"/readme",children:(0,l.jsx)("div",{css:h(s,"readme"==a),children:"Readme"})}),(0,l.jsx)("div",{css:{flexGrow:1}}),m&&!1,m&&!1,(0,l.jsx)("div",{css:{width:"100px"}}),!1,(0,l.jsxs)("div",{css:(n=p,(0,r.iv)({fontSize:"13px",color:n?"lime":"red",marginRight:"50px",display:"flex",flexDirection:"row",alignItems:"center"})),children:[(0,l.jsx)(d,{iconStyle:function(){return x()},color:p?"lime":"red",children:p?"wifi_on":"wifi_off"}),"Model "+(p?"connected":"disconnected")]})]})}},1281:function(e,n,t){"use strict";t.d(n,{Z:function(){return d}});var i=t(7294),r=t(9008),o=t(950),c=t(3693),s=t(5893),l=function(e){return{marginTop:"100px",marginLeft:"10px",marginRight:"10px",padding:"20px",borderRadius:"10px",backgroundColor:e.colors.primary,color:e.colors.blond,width:"fit-content",height:"fit-content",textAlign:"center",lineHeight:"1.7em"}},a=function(e){var n=(0,o.H)().style;return(0,s.jsx)(c.Z,{style:l(n),children:"Mobile version not ready. Use your PC."})},d=function(e){var n=(0,o.H)(),t=n.style,c=n.device;return(0,i.useEffect)((function(){c.setIsMobile(Boolean(navigator.userAgent.match(/Android|BlackBerry|iPhone|iPad|iPod|Opera Mini|IEMobile|WPDesktop/i)))}),[]),(0,s.jsxs)("div",{css:t.pageS,children:[(0,s.jsxs)(r.default,{children:[(0,s.jsx)("title",{children:e.title}),(0,s.jsx)("meta",{name:"description",content:e.description}),(0,s.jsx)("link",{rel:"icon",href:"/favicon.ico"}),(0,s.jsx)("link",{href:"https://fonts.googleapis.com/icon?family=Material+Icons",rel:"stylesheet"})]}),!c.isMobile&&e.children,c.isMobile&&(0,s.jsx)(a,{})]})}},3077:function(e,n,t){"use strict";t.r(n);var i=t(266),r=t(809),o=t.n(r),c=t(917),s=t(7294),l=t(2903),a=t(1281),d=t(4280),u=t(950),f=t(5893),h=function(e){return(0,c.iv)({width:"80%",textAlign:"left",marginTop:"30px",height:"calc(100vh - ".concat(e.view.headerHeight,")"),overflowY:"scroll",lineHeight:"1.7em"})};n.default=function(){var e=(0,u.H)(),n=e.setPage,t=e.style,r=(0,s.useState)(""),c=r[0],x=r[1],g=function(){var e=(0,i.Z)(o().mark((function e(){var n;return o().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,fetch("/data/readme.md",{method:"GET"});case 2:return n=e.sent,e.t0=x,e.next=6,n.text();case 6:e.t1=e.sent,(0,e.t0)(e.t1);case 8:case"end":return e.stop()}}),e)})));return function(){return e.apply(this,arguments)}}();return(0,s.useEffect)((function(){n("readme"),g()}),[]),(0,f.jsxs)(a.Z,{title:"KKY | MDE Corpus | Readme",description:"Data Analysis",children:[(0,f.jsx)(d.Z,{}),(0,f.jsx)("div",{css:h(t),children:(0,f.jsx)(l.D,{children:c})})]})}},7339:function(e,n,t){(window.__NEXT_P=window.__NEXT_P||[]).push(["/readme",function(){return t(3077)}])}},function(e){e.O(0,[774,19,903,888,179],(function(){return n=7339,e(e.s=n);var n}));var n=e.O();_N_E=n}]);