(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[888],{1859:function(e,t,n){"use strict";n.d(t,{Z:function(){return ne}});var r=n(1526),o=Math.abs,c=String.fromCharCode;function a(e){return e.trim()}function s(e,t,n){return e.replace(t,n)}function i(e,t){return e.indexOf(t)}function u(e,t){return 0|e.charCodeAt(t)}function f(e,t,n){return e.slice(t,n)}function l(e){return e.length}function p(e){return e.length}function d(e,t){return t.push(e),e}var h=1,y=1,v=0,g=0,m=0,b="";function w(e,t,n,r,o,c,a){return{value:e,root:t,parent:n,type:r,props:o,children:c,line:h,column:y,length:a,return:""}}function x(e,t,n){return w(e,t.root,t.parent,n,t.props,t.children,0)}function S(){return m=g>0?u(b,--g):0,y--,10===m&&(y=1,h--),m}function k(){return m=g<v?u(b,g++):0,y++,10===m&&(y=1,h++),m}function $(){return u(b,g)}function C(){return g}function O(e,t){return f(b,e,t)}function _(e){switch(e){case 0:case 9:case 10:case 13:case 32:return 5;case 33:case 43:case 44:case 47:case 62:case 64:case 126:case 59:case 123:case 125:return 4;case 58:return 3;case 34:case 39:case 40:case 91:return 2;case 41:case 93:return 1}return 0}function j(e){return h=y=1,v=l(b=e),g=0,[]}function P(e){return b="",e}function A(e){return a(O(g-1,T(91===e?e+2:40===e?e+1:e)))}function E(e){for(;(m=$())&&m<33;)k();return _(e)>2||_(m)>3?"":" "}function z(e,t){for(;--t&&k()&&!(m<48||m>102||m>57&&m<65||m>70&&m<97););return O(e,C()+(t<6&&32==$()&&32==k()))}function T(e){for(;k();)switch(m){case e:return g;case 34:case 39:return T(34===e||39===e?e:m);case 40:41===e&&T(e);break;case 92:k()}return g}function M(e,t){for(;k()&&e+m!==57&&(e+m!==84||47!==$()););return"/*"+O(t,g-1)+"*"+c(47===e?e:k())}function Z(e){for(;!_($());)k();return O(e,g)}var N="-ms-",R="-moz-",L="-webkit-",D="comm",F="rule",I="decl";function G(e,t){for(var n="",r=p(e),o=0;o<r;o++)n+=t(e[o],o,e,t)||"";return n}function H(e,t,n,r){switch(e.type){case"@import":case I:return e.return=e.return||e.value;case D:return"";case F:e.value=e.props.join(",")}return l(n=G(e.children,r))?e.return=e.value+"{"+n+"}":""}function q(e,t){switch(function(e,t){return(((t<<2^u(e,0))<<2^u(e,1))<<2^u(e,2))<<2^u(e,3)}(e,t)){case 5103:return L+"print-"+e+e;case 5737:case 4201:case 3177:case 3433:case 1641:case 4457:case 2921:case 5572:case 6356:case 5844:case 3191:case 6645:case 3005:case 6391:case 5879:case 5623:case 6135:case 4599:case 4855:case 4215:case 6389:case 5109:case 5365:case 5621:case 3829:return L+e+e;case 5349:case 4246:case 4810:case 6968:case 2756:return L+e+R+e+N+e+e;case 6828:case 4268:return L+e+N+e+e;case 6165:return L+e+N+"flex-"+e+e;case 5187:return L+e+s(e,/(\w+).+(:[^]+)/,"-webkit-box-$1$2-ms-flex-$1$2")+e;case 5443:return L+e+N+"flex-item-"+s(e,/flex-|-self/,"")+e;case 4675:return L+e+N+"flex-line-pack"+s(e,/align-content|flex-|-self/,"")+e;case 5548:return L+e+N+s(e,"shrink","negative")+e;case 5292:return L+e+N+s(e,"basis","preferred-size")+e;case 6060:return L+"box-"+s(e,"-grow","")+L+e+N+s(e,"grow","positive")+e;case 4554:return L+s(e,/([^-])(transform)/g,"$1-webkit-$2")+e;case 6187:return s(s(s(e,/(zoom-|grab)/,L+"$1"),/(image-set)/,L+"$1"),e,"")+e;case 5495:case 3959:return s(e,/(image-set\([^]*)/,L+"$1$`$1");case 4968:return s(s(e,/(.+:)(flex-)?(.*)/,"-webkit-box-pack:$3-ms-flex-pack:$3"),/s.+-b[^;]+/,"justify")+L+e+e;case 4095:case 3583:case 4068:case 2532:return s(e,/(.+)-inline(.+)/,L+"$1$2")+e;case 8116:case 7059:case 5753:case 5535:case 5445:case 5701:case 4933:case 4677:case 5533:case 5789:case 5021:case 4765:if(l(e)-1-t>6)switch(u(e,t+1)){case 109:if(45!==u(e,t+4))break;case 102:return s(e,/(.+:)(.+)-([^]+)/,"$1-webkit-$2-$3$1"+R+(108==u(e,t+3)?"$3":"$2-$3"))+e;case 115:return~i(e,"stretch")?q(s(e,"stretch","fill-available"),t)+e:e}break;case 4949:if(115!==u(e,t+1))break;case 6444:switch(u(e,l(e)-3-(~i(e,"!important")&&10))){case 107:return s(e,":",":"+L)+e;case 101:return s(e,/(.+:)([^;!]+)(;|!.+)?/,"$1"+L+(45===u(e,14)?"inline-":"")+"box$3$1"+L+"$2$3$1"+N+"$2box$3")+e}break;case 5936:switch(u(e,t+11)){case 114:return L+e+N+s(e,/[svh]\w+-[tblr]{2}/,"tb")+e;case 108:return L+e+N+s(e,/[svh]\w+-[tblr]{2}/,"tb-rl")+e;case 45:return L+e+N+s(e,/[svh]\w+-[tblr]{2}/,"lr")+e}return L+e+N+e+e}return e}function W(e){return P(B("",null,null,null,[""],e=j(e),0,[0],e))}function B(e,t,n,r,o,a,i,u,f){for(var p=0,h=0,y=i,v=0,g=0,m=0,b=1,w=1,x=1,O=0,_="",j=o,P=a,T=r,N=_;w;)switch(m=O,O=k()){case 34:case 39:case 91:case 40:N+=A(O);break;case 9:case 10:case 13:case 32:N+=E(m);break;case 92:N+=z(C()-1,7);continue;case 47:switch($()){case 42:case 47:d(J(M(k(),C()),t,n),f);break;default:N+="/"}break;case 123*b:u[p++]=l(N)*x;case 125*b:case 59:case 0:switch(O){case 0:case 125:w=0;case 59+h:g>0&&l(N)-y&&d(g>32?X(N+";",r,n,y-1):X(s(N," ","")+";",r,n,y-2),f);break;case 59:N+=";";default:if(d(T=U(N,t,n,p,h,o,u,_,j=[],P=[],y),a),123===O)if(0===h)B(N,t,T,T,j,a,y,u,P);else switch(v){case 100:case 109:case 115:B(e,T,T,r&&d(U(e,T,T,0,0,o,u,_,o,j=[],y),P),o,P,y,u,r?j:P);break;default:B(N,T,T,T,[""],P,y,u,P)}}p=h=g=0,b=x=1,_=N="",y=i;break;case 58:y=1+l(N),g=m;default:if(b<1)if(123==O)--b;else if(125==O&&0==b++&&125==S())continue;switch(N+=c(O),O*b){case 38:x=h>0?1:(N+="\f",-1);break;case 44:u[p++]=(l(N)-1)*x,x=1;break;case 64:45===$()&&(N+=A(k())),v=$(),h=l(_=N+=Z(C())),O++;break;case 45:45===m&&2==l(N)&&(b=0)}}return a}function U(e,t,n,r,c,i,u,l,d,h,y){for(var v=c-1,g=0===c?i:[""],m=p(g),b=0,x=0,S=0;b<r;++b)for(var k=0,$=f(e,v+1,v=o(x=u[b])),C=e;k<m;++k)(C=a(x>0?g[k]+" "+$:s($,/&\f/g,g[k])))&&(d[S++]=C);return w(e,t,n,0===c?F:l,d,h,y)}function J(e,t,n){return w(e,t,n,D,c(m),f(e,2,-2),0)}function X(e,t,n,r){return w(e,t,n,I,f(e,0,r),f(e,r+1,-1),r)}var V=function(e,t,n){for(var r=0,o=0;r=o,o=$(),38===r&&12===o&&(t[n]=1),!_(o);)k();return O(e,g)},K=function(e,t){return P(function(e,t){var n=-1,r=44;do{switch(_(r)){case 0:38===r&&12===$()&&(t[n]=1),e[n]+=V(g-1,t,n);break;case 2:e[n]+=A(r);break;case 4:if(44===r){e[++n]=58===$()?"&\f":"",t[n]=e[n].length;break}default:e[n]+=c(r)}}while(r=k());return e}(j(e),t))},Q=new WeakMap,Y=function(e){if("rule"===e.type&&e.parent&&e.length){for(var t=e.value,n=e.parent,r=e.column===n.column&&e.line===n.line;"rule"!==n.type;)if(!(n=n.parent))return;if((1!==e.props.length||58===t.charCodeAt(0)||Q.get(n))&&!r){Q.set(e,!0);for(var o=[],c=K(t,o),a=n.props,s=0,i=0;s<c.length;s++)for(var u=0;u<a.length;u++,i++)e.props[i]=o[s]?c[s].replace(/&\f/g,a[u]):a[u]+" "+c[s]}}},ee=function(e){if("decl"===e.type){var t=e.value;108===t.charCodeAt(0)&&98===t.charCodeAt(2)&&(e.return="",e.value="")}},te=[function(e,t,n,r){if(!e.return)switch(e.type){case I:e.return=q(e.value,e.length);break;case"@keyframes":return G([x(s(e.value,"@","@"+L),e,"")],r);case F:if(e.length)return function(e,t){return e.map(t).join("")}(e.props,(function(t){switch(function(e,t){return(e=t.exec(e))?e[0]:e}(t,/(::plac\w+|:read-\w+)/)){case":read-only":case":read-write":return G([x(s(t,/:(read-\w+)/,":-moz-$1"),e,"")],r);case"::placeholder":return G([x(s(t,/:(plac\w+)/,":-webkit-input-$1"),e,""),x(s(t,/:(plac\w+)/,":-moz-$1"),e,""),x(s(t,/:(plac\w+)/,N+"input-$1"),e,"")],r)}return""}))}}],ne=function(e){var t=e.key;if("css"===t){var n=document.querySelectorAll("style[data-emotion]:not([data-s])");Array.prototype.forEach.call(n,(function(e){-1!==e.getAttribute("data-emotion").indexOf(" ")&&(document.head.appendChild(e),e.setAttribute("data-s",""))}))}var o=e.stylisPlugins||te;var c,a,s={},i=[];c=e.container||document.head,Array.prototype.forEach.call(document.querySelectorAll('style[data-emotion^="'+t+' "]'),(function(e){for(var t=e.getAttribute("data-emotion").split(" "),n=1;n<t.length;n++)s[t[n]]=!0;i.push(e)}));var u,f,l=[H,(f=function(e){u.insert(e)},function(e){e.root||(e=e.return)&&f(e)})],d=function(e){var t=p(e);return function(n,r,o,c){for(var a="",s=0;s<t;s++)a+=e[s](n,r,o,c)||"";return a}}([Y,ee].concat(o,l));a=function(e,t,n,r){u=n,G(W(e?e+"{"+t.styles+"}":t.styles),d),r&&(h.inserted[t.name]=!0)};var h={key:t,sheet:new r.m({key:t,container:c,nonce:e.nonce,speedy:e.speedy,prepend:e.prepend}),nonce:e.nonce,inserted:s,registered:{},insert:a};return h.sheet.hydrate(i),h}},7866:function(e,t){"use strict";t.Z=function(e){var t=Object.create(null);return function(n){return void 0===t[n]&&(t[n]=e(n)),t[n]}}},4759:function(e,t,n){"use strict";n.d(t,{T:function(){return s},w:function(){return a}});var r=n(7294),o=n(1859),c=(n(4199),Object.prototype.hasOwnProperty,(0,r.createContext)("undefined"!==typeof HTMLElement?(0,o.Z)({key:"css"}):null));c.Provider;var a=function(e){return(0,r.forwardRef)((function(t,n){var o=(0,r.useContext)(c);return e(t,o,n)}))},s=(0,r.createContext)({})},917:function(e,t,n){"use strict";n.d(t,{xB:function(){return i},iv:function(){return u}});var r=n(7294),o=(n(1859),n(4759)),c=(n(8679),n(444)),a=n(4199),s=n(1526),i=(0,o.w)((function(e,t){var n=e.styles,i=(0,a.O)([n],void 0,(0,r.useContext)(o.T)),u=(0,r.useRef)();return(0,r.useLayoutEffect)((function(){var e=t.key+"-global",n=new s.m({key:e,nonce:t.sheet.nonce,container:t.sheet.container,speedy:t.sheet.isSpeedy}),r=!1,o=document.querySelector('style[data-emotion="'+e+" "+i.name+'"]');return t.sheet.tags.length&&(n.before=t.sheet.tags[0]),null!==o&&(r=!0,o.setAttribute("data-emotion",e),n.hydrate([o])),u.current=[n,r],function(){n.flush()}}),[t]),(0,r.useLayoutEffect)((function(){var e=u.current,n=e[0];if(e[1])e[1]=!1;else{if(void 0!==i.next&&(0,c.M)(t,i.next,!0),n.tags.length){var r=n.tags[n.tags.length-1].nextElementSibling;n.before=r,n.flush()}t.insert("",i,n,!1)}}),[t,i.name]),null}));function u(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return(0,a.O)(t)}},4199:function(e,t,n){"use strict";n.d(t,{O:function(){return y}});var r=function(e){for(var t,n=0,r=0,o=e.length;o>=4;++r,o-=4)t=1540483477*(65535&(t=255&e.charCodeAt(r)|(255&e.charCodeAt(++r))<<8|(255&e.charCodeAt(++r))<<16|(255&e.charCodeAt(++r))<<24))+(59797*(t>>>16)<<16),n=1540483477*(65535&(t^=t>>>24))+(59797*(t>>>16)<<16)^1540483477*(65535&n)+(59797*(n>>>16)<<16);switch(o){case 3:n^=(255&e.charCodeAt(r+2))<<16;case 2:n^=(255&e.charCodeAt(r+1))<<8;case 1:n=1540483477*(65535&(n^=255&e.charCodeAt(r)))+(59797*(n>>>16)<<16)}return(((n=1540483477*(65535&(n^=n>>>13))+(59797*(n>>>16)<<16))^n>>>15)>>>0).toString(36)},o={animationIterationCount:1,borderImageOutset:1,borderImageSlice:1,borderImageWidth:1,boxFlex:1,boxFlexGroup:1,boxOrdinalGroup:1,columnCount:1,columns:1,flex:1,flexGrow:1,flexPositive:1,flexShrink:1,flexNegative:1,flexOrder:1,gridRow:1,gridRowEnd:1,gridRowSpan:1,gridRowStart:1,gridColumn:1,gridColumnEnd:1,gridColumnSpan:1,gridColumnStart:1,msGridRow:1,msGridRowSpan:1,msGridColumn:1,msGridColumnSpan:1,fontWeight:1,lineHeight:1,opacity:1,order:1,orphans:1,tabSize:1,widows:1,zIndex:1,zoom:1,WebkitLineClamp:1,fillOpacity:1,floodOpacity:1,stopOpacity:1,strokeDasharray:1,strokeDashoffset:1,strokeMiterlimit:1,strokeOpacity:1,strokeWidth:1},c=n(7866),a=/[A-Z]|^ms/g,s=/_EMO_([^_]+?)_([^]*?)_EMO_/g,i=function(e){return 45===e.charCodeAt(1)},u=function(e){return null!=e&&"boolean"!==typeof e},f=(0,c.Z)((function(e){return i(e)?e:e.replace(a,"-$&").toLowerCase()})),l=function(e,t){switch(e){case"animation":case"animationName":if("string"===typeof t)return t.replace(s,(function(e,t,n){return d={name:t,styles:n,next:d},t}))}return 1===o[e]||i(e)||"number"!==typeof t||0===t?t:t+"px"};function p(e,t,n){if(null==n)return"";if(void 0!==n.__emotion_styles)return n;switch(typeof n){case"boolean":return"";case"object":if(1===n.anim)return d={name:n.name,styles:n.styles,next:d},n.name;if(void 0!==n.styles){var r=n.next;if(void 0!==r)for(;void 0!==r;)d={name:r.name,styles:r.styles,next:d},r=r.next;return n.styles+";"}return function(e,t,n){var r="";if(Array.isArray(n))for(var o=0;o<n.length;o++)r+=p(e,t,n[o])+";";else for(var c in n){var a=n[c];if("object"!==typeof a)null!=t&&void 0!==t[a]?r+=c+"{"+t[a]+"}":u(a)&&(r+=f(c)+":"+l(c,a)+";");else if(!Array.isArray(a)||"string"!==typeof a[0]||null!=t&&void 0!==t[a[0]]){var s=p(e,t,a);switch(c){case"animation":case"animationName":r+=f(c)+":"+s+";";break;default:r+=c+"{"+s+"}"}}else for(var i=0;i<a.length;i++)u(a[i])&&(r+=f(c)+":"+l(c,a[i])+";")}return r}(e,t,n);case"function":if(void 0!==e){var o=d,c=n(e);return d=o,p(e,t,c)}break;case"string":}if(null==t)return n;var a=t[n];return void 0!==a?a:n}var d,h=/label:\s*([^\s;\n{]+)\s*(;|$)/g;var y=function(e,t,n){if(1===e.length&&"object"===typeof e[0]&&null!==e[0]&&void 0!==e[0].styles)return e[0];var o=!0,c="";d=void 0;var a=e[0];null==a||void 0===a.raw?(o=!1,c+=p(n,t,a)):c+=a[0];for(var s=1;s<e.length;s++)c+=p(n,t,e[s]),o&&(c+=a[s]);h.lastIndex=0;for(var i,u="";null!==(i=h.exec(c));)u+="-"+i[1];return{name:r(c)+u,styles:c,next:d}}},1526:function(e,t,n){"use strict";n.d(t,{m:function(){return r}});var r=function(){function e(e){var t=this;this._insertTag=function(e){var n;n=0===t.tags.length?t.prepend?t.container.firstChild:t.before:t.tags[t.tags.length-1].nextSibling,t.container.insertBefore(e,n),t.tags.push(e)},this.isSpeedy=void 0===e.speedy||e.speedy,this.tags=[],this.ctr=0,this.nonce=e.nonce,this.key=e.key,this.container=e.container,this.prepend=e.prepend,this.before=null}var t=e.prototype;return t.hydrate=function(e){e.forEach(this._insertTag)},t.insert=function(e){this.ctr%(this.isSpeedy?65e3:1)===0&&this._insertTag(function(e){var t=document.createElement("style");return t.setAttribute("data-emotion",e.key),void 0!==e.nonce&&t.setAttribute("nonce",e.nonce),t.appendChild(document.createTextNode("")),t.setAttribute("data-s",""),t}(this));var t=this.tags[this.tags.length-1];if(this.isSpeedy){var n=function(e){if(e.sheet)return e.sheet;for(var t=0;t<document.styleSheets.length;t++)if(document.styleSheets[t].ownerNode===e)return document.styleSheets[t]}(t);try{n.insertRule(e,n.cssRules.length)}catch(r){0}}else t.appendChild(document.createTextNode(e));this.ctr++},t.flush=function(){this.tags.forEach((function(e){return e.parentNode&&e.parentNode.removeChild(e)})),this.tags=[],this.ctr=0},e}()},444:function(e,t,n){"use strict";n.d(t,{f:function(){return r},M:function(){return o}});function r(e,t,n){var r="";return n.split(" ").forEach((function(n){void 0!==e[n]?t.push(e[n]+";"):r+=n+" "})),r}var o=function(e,t,n){var r=e.key+"-"+t.name;if(!1===n&&void 0===e.registered[r]&&(e.registered[r]=t.styles),void 0===e.inserted[t.name]){var o=t;do{e.insert(t===o?"."+r:"",o,e.sheet,!0);o=o.next}while(void 0!==o)}}},8679:function(e,t,n){"use strict";var r=n(1296),o={childContextTypes:!0,contextType:!0,contextTypes:!0,defaultProps:!0,displayName:!0,getDefaultProps:!0,getDerivedStateFromError:!0,getDerivedStateFromProps:!0,mixins:!0,propTypes:!0,type:!0},c={name:!0,length:!0,prototype:!0,caller:!0,callee:!0,arguments:!0,arity:!0},a={$$typeof:!0,compare:!0,defaultProps:!0,displayName:!0,propTypes:!0,type:!0},s={};function i(e){return r.isMemo(e)?a:s[e.$$typeof]||o}s[r.ForwardRef]={$$typeof:!0,render:!0,defaultProps:!0,displayName:!0,propTypes:!0},s[r.Memo]=a;var u=Object.defineProperty,f=Object.getOwnPropertyNames,l=Object.getOwnPropertySymbols,p=Object.getOwnPropertyDescriptor,d=Object.getPrototypeOf,h=Object.prototype;e.exports=function e(t,n,r){if("string"!==typeof n){if(h){var o=d(n);o&&o!==h&&e(t,o,r)}var a=f(n);l&&(a=a.concat(l(n)));for(var s=i(t),y=i(n),v=0;v<a.length;++v){var g=a[v];if(!c[g]&&(!r||!r[g])&&(!y||!y[g])&&(!s||!s[g])){var m=p(n,g);try{u(t,g,m)}catch(b){}}}}return t}},6103:function(e,t){"use strict";var n="function"===typeof Symbol&&Symbol.for,r=n?Symbol.for("react.element"):60103,o=n?Symbol.for("react.portal"):60106,c=n?Symbol.for("react.fragment"):60107,a=n?Symbol.for("react.strict_mode"):60108,s=n?Symbol.for("react.profiler"):60114,i=n?Symbol.for("react.provider"):60109,u=n?Symbol.for("react.context"):60110,f=n?Symbol.for("react.async_mode"):60111,l=n?Symbol.for("react.concurrent_mode"):60111,p=n?Symbol.for("react.forward_ref"):60112,d=n?Symbol.for("react.suspense"):60113,h=n?Symbol.for("react.suspense_list"):60120,y=n?Symbol.for("react.memo"):60115,v=n?Symbol.for("react.lazy"):60116,g=n?Symbol.for("react.block"):60121,m=n?Symbol.for("react.fundamental"):60117,b=n?Symbol.for("react.responder"):60118,w=n?Symbol.for("react.scope"):60119;function x(e){if("object"===typeof e&&null!==e){var t=e.$$typeof;switch(t){case r:switch(e=e.type){case f:case l:case c:case s:case a:case d:return e;default:switch(e=e&&e.$$typeof){case u:case p:case v:case y:case i:return e;default:return t}}case o:return t}}}function S(e){return x(e)===l}t.AsyncMode=f,t.ConcurrentMode=l,t.ContextConsumer=u,t.ContextProvider=i,t.Element=r,t.ForwardRef=p,t.Fragment=c,t.Lazy=v,t.Memo=y,t.Portal=o,t.Profiler=s,t.StrictMode=a,t.Suspense=d,t.isAsyncMode=function(e){return S(e)||x(e)===f},t.isConcurrentMode=S,t.isContextConsumer=function(e){return x(e)===u},t.isContextProvider=function(e){return x(e)===i},t.isElement=function(e){return"object"===typeof e&&null!==e&&e.$$typeof===r},t.isForwardRef=function(e){return x(e)===p},t.isFragment=function(e){return x(e)===c},t.isLazy=function(e){return x(e)===v},t.isMemo=function(e){return x(e)===y},t.isPortal=function(e){return x(e)===o},t.isProfiler=function(e){return x(e)===s},t.isStrictMode=function(e){return x(e)===a},t.isSuspense=function(e){return x(e)===d},t.isValidElementType=function(e){return"string"===typeof e||"function"===typeof e||e===c||e===l||e===s||e===a||e===d||e===h||"object"===typeof e&&null!==e&&(e.$$typeof===v||e.$$typeof===y||e.$$typeof===i||e.$$typeof===u||e.$$typeof===p||e.$$typeof===m||e.$$typeof===b||e.$$typeof===w||e.$$typeof===g)},t.typeOf=x},1296:function(e,t,n){"use strict";e.exports=n(6103)},1770:function(e,t,n){"use strict";n.d(t,{L:function(){return c}});var r="localhost",o="6841",c={uri_log:"http://"+r+":"+o+"/log/",uri_query:"http://"+r+":"+o+"/query/",sc:{useSc:!1,libSrc:"https://cak.zcu.cz:9444/speechcloud.js",options:{uri:"https://prod.speechcloud.kky.zcu.cz:9443/v1/speechcloud/devel/bulinm/robot-v1-cs",tts:"#audioout",disable_audio_processing:!0}}}},5077:function(e,t,n){"use strict";n.d(t,{f:function(){return c}});var r=n(7294),o=(0,r.createContext)({}),c=function(){return(0,r.useContext)(o)};t.Z=o},950:function(e,t,n){"use strict";n.d(t,{H:function(){return c}});var r=n(7294),o=(0,r.createContext)({}),c=function(){return(0,r.useContext)(o)};t.Z=o},1164:function(e,t,n){"use strict";n.r(t),n.d(t,{default:function(){return j}});var r,o=n(2809),c=n(917),a=n(950),s=n(5077),i=n(7294);var u,f,l={primary:"#00008b",secondary:"#eeeeee",blond:"#ffffff",dark:"#000000",highlight:"#f7edad",gray:"#dddddd",inverse:"#800000",green:"#008000",darkgray:"#aaaaaa"},p={headerHeight:"50px"},d=(0,c.iv)({display:"flex",flexDirection:"column",minWidth:"100vh",minHeight:"100vh",maxHeight:"100vh",overflow:"hidden",alignItems:"center",justifyContent:"flex-start"}),h=(0,c.iv)(r||(u=["\nhtml {\n    background: black;\n  }\n  body {\n    min-width: 100vh;\n    min-height: 100vh;\n    margin: 0 auto;\n    background: ",";\n    font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue, sans-serif;\n  }\n\na {\n  color: inherit;\n  text-decoration: none;\n}\n"],f||(f=u.slice(0)),r=Object.freeze(Object.defineProperties(u,{raw:{value:Object.freeze(f)}}))),l.secondary),y=function(){var e=(0,i.useState)("systems"),t=e[0],n=e[1],r=(0,i.useState)(!1),o=r[0],c=r[1];return{style:{globalStyle:h,pageS:d,colors:l,view:p},device:{isMobile:o,setIsMobile:c},page:t,setPage:n}},v=n(266),g=n(809),m=n.n(g),b=function(){var e=(0,v.Z)(m().mark((function e(t){var n;return m().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,fetch(t,{method:"GET"});case 2:return n=e.sent,e.next=5,n.json();case 5:return e.abrupt("return",e.sent);case 6:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}(),w=function(){var e=(0,v.Z)(m().mark((function e(t,n){var r;return m().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,fetch(t,{body:JSON.stringify(n),method:"POST"});case 2:return r=e.sent,e.next=5,r.json();case 5:return e.abrupt("return",e.sent);case 6:case"end":return e.stop()}}),e)})));return function(t,n){return e.apply(this,arguments)}}(),x=n(1770),S=function(){var e=(0,i.useState)(!1),t=e[0],n=e[1],r=(0,i.useState)([]),o=r[0],c=r[1],a=(0,i.useState)(!1),s=a[0],u=a[1],f=(0,i.useState)(!1),l=f[0],p=f[1],d=(0,i.useState)(!0),h=d[0],y=d[1],g=(0,i.useState)({}),S=g[0],k=g[1],$=(0,i.useState)(!1),C=$[0],O=$[1],_=(0,i.useState)(!1),j=_[0],P=_[1],A=(0,i.useState)(!1),E=A[0],z=A[1],T=(0,i.useState)(!1),M=T[0],Z=(T[1],function(){var e=(0,v.Z)(m().mark((function e(){return m().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(t){e.next=3;break}return e.next=3,b(x.L.uri_log).then(function(){var e=(0,v.Z)(m().mark((function e(t){return m().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:n(t.online),console.log("Logged. Server online.",t);case 2:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}()).catch((function(){n(!1),console.log("Server not online")}));case 3:case"end":return e.stop()}}),e)})));return function(){return e.apply(this,arguments)}}()),N=function(){var e=(0,v.Z)(m().mark((function e(n){return m().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(!t){e.next=5;break}return e.next=3,L(n).then(function(){var e=(0,v.Z)(m().mark((function e(t){return m().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return console.log("human message set:",t),e.next=3,w(x.L.uri_query,{message:n}).then(function(){var e=(0,v.Z)(m().mark((function e(n){return m().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:console.log("Got result:",n),D(n,t);case 2:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}()).catch((function(){console.log("E: Unable to fetch Env data.")}));case 3:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}());case 3:e.next=6;break;case 5:console.log("E: Not connected.");case 6:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}(),R=function(){var e=new Date;return e.toLocaleString("cs")+"."+e.getMilliseconds()},L=function(){var e=(0,v.Z)(m().mark((function e(t){var n,r;return m().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return u(!0),n={text:t,source:"human",time:R(),dt:0},console.log("Human appending to",o),r=o.concat([n]),c(r),e.abrupt("return",r);case 6:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}(),D=function(){var e=(0,v.Z)(m().mark((function e(t,n){var r;return m().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:u(!1),r={text:t.text,source:"bot",time:R(),dt:t.dt},console.log("Bot appending to",n),c(n.concat([r]));case 4:case"end":return e.stop()}}),e)})));return function(t,n){return e.apply(this,arguments)}}(),F=function(){console.log("SC: ws connected.")},I=function(){P(!1),z(!1),console.log("SC: ws closed.")},G=function(){console.log("SC: ws session.")},H=function(){P(!0),console.log("SC: ASR ready.")},q=function(){z(!0),console.log("SC: Recognizing...")},W=function(e){z(!1),console.log("SC: ASR result.",e)},B=function(e){P(!1),z(!1),console.log("SC: Error.",e)},U={usingASR:l,setUsingASR:p,usingTTS:h,setUsingTTS:y,scLib:S,initialized:C,init:function(){var e=(0,v.Z)(m().mark((function e(){var t;return m().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return C&&O(!1),console.log("Initializing SC..."),e.next=4,new SpeechCloud(x.L.sc.options);case 4:t=e.sent,console.log("Got instance:",t),console.log("Listeners registration..."),t.on("_ws_connected",F),t.on("_ws_closed",I),t.on("_ws_session",G),t.on("asr_ready",H),t.on("asr_recognizing",q),t.on("asr_result",W),t.on("sc_error",B),t.init(),console.log("SC initialized."),console.log(t),k(t),O(!0);case 19:case"end":return e.stop()}}),e)})));return function(){return e.apply(this,arguments)}}(),asrReady:j,recognize:function(){S&&!E&&S.asr_recognize()},recognizing:E,synthesize:function(e){console.log("Trying to synthesize:",e),S._ws.send(JSON.stringify({type:"tts_synthesize",text:e})),console.log("Synthesized.")},synthesizing:M};return{logged:t,log:Z,sendToServer:N,dialog:o,robotThinking:s,sc:U}},k=n(4298),$=n(5893),C=function(){var e=(0,s.f)().sc;return(0,$.jsx)(i.Fragment,{children:x.L.sc.useSc&&(0,$.jsx)(k.default,{id:"scScript",src:x.L.sc.libSrc,strategy:"afterInteractive",onLoad:function(){return e.init()}})})};function O(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function _(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?O(Object(n),!0).forEach((function(t){(0,o.Z)(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):O(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}var j=function(e){var t=e.Component,n=e.pageProps,r=y(),o=S();return(0,i.useEffect)((function(){o.logged||o.log()}),[]),(0,$.jsx)(a.Z.Provider,{value:r,children:(0,$.jsxs)(s.Z.Provider,{value:o,children:[(0,$.jsx)(c.xB,{styles:r.style.globalStyle}),(0,$.jsx)(C,{}),(0,$.jsx)(t,_({},n))]})})}},6363:function(e,t,n){(window.__NEXT_P=window.__NEXT_P||[]).push(["/_app",function(){return n(1164)}])},4298:function(e,t,n){e.exports=n(7926)},266:function(e,t,n){"use strict";function r(e,t,n,r,o,c,a){try{var s=e[c](a),i=s.value}catch(u){return void n(u)}s.done?t(i):Promise.resolve(i).then(r,o)}function o(e){return function(){var t=this,n=arguments;return new Promise((function(o,c){var a=e.apply(t,n);function s(e){r(a,o,c,s,i,"next",e)}function i(e){r(a,o,c,s,i,"throw",e)}s(void 0)}))}}n.d(t,{Z:function(){return o}})},2809:function(e,t,n){"use strict";function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}n.d(t,{Z:function(){return r}})}},function(e){var t=function(t){return e(e.s=t)};e.O(0,[774,179],(function(){return t(6363),t(4651)}));var n=e.O();_N_E=n}]);