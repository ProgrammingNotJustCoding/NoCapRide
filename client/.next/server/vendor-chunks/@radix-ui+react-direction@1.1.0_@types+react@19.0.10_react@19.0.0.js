"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
exports.id = "vendor-chunks/@radix-ui+react-direction@1.1.0_@types+react@19.0.10_react@19.0.0";
exports.ids = ["vendor-chunks/@radix-ui+react-direction@1.1.0_@types+react@19.0.10_react@19.0.0"];
exports.modules = {

/***/ "(ssr)/./node_modules/.pnpm/@radix-ui+react-direction@1.1.0_@types+react@19.0.10_react@19.0.0/node_modules/@radix-ui/react-direction/dist/index.mjs":
/*!****************************************************************************************************************************************************!*\
  !*** ./node_modules/.pnpm/@radix-ui+react-direction@1.1.0_@types+react@19.0.10_react@19.0.0/node_modules/@radix-ui/react-direction/dist/index.mjs ***!
  \****************************************************************************************************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   DirectionProvider: () => (/* binding */ DirectionProvider),\n/* harmony export */   Provider: () => (/* binding */ Provider),\n/* harmony export */   useDirection: () => (/* binding */ useDirection)\n/* harmony export */ });\n/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ \"(ssr)/./node_modules/.pnpm/next@15.2.2_react-dom@19.0.0_react@19.0.0__react@19.0.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js\");\n/* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react/jsx-runtime */ \"(ssr)/./node_modules/.pnpm/next@15.2.2_react-dom@19.0.0_react@19.0.0__react@19.0.0/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-runtime.js\");\n// packages/react/direction/src/Direction.tsx\n\n\nvar DirectionContext = react__WEBPACK_IMPORTED_MODULE_0__.createContext(void 0);\nvar DirectionProvider = (props) => {\n  const { dir, children } = props;\n  return /* @__PURE__ */ (0,react_jsx_runtime__WEBPACK_IMPORTED_MODULE_1__.jsx)(DirectionContext.Provider, { value: dir, children });\n};\nfunction useDirection(localDir) {\n  const globalDir = react__WEBPACK_IMPORTED_MODULE_0__.useContext(DirectionContext);\n  return localDir || globalDir || \"ltr\";\n}\nvar Provider = DirectionProvider;\n\n//# sourceMappingURL=index.mjs.map\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKHNzcikvLi9ub2RlX21vZHVsZXMvLnBucG0vQHJhZGl4LXVpK3JlYWN0LWRpcmVjdGlvbkAxLjEuMF9AdHlwZXMrcmVhY3RAMTkuMC4xMF9yZWFjdEAxOS4wLjAvbm9kZV9tb2R1bGVzL0ByYWRpeC11aS9yZWFjdC1kaXJlY3Rpb24vZGlzdC9pbmRleC5tanMiLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7QUFBQTtBQUMrQjtBQUNTO0FBQ3hDLHVCQUF1QixnREFBbUI7QUFDMUM7QUFDQSxVQUFVLGdCQUFnQjtBQUMxQix5QkFBeUIsc0RBQUcsOEJBQThCLHNCQUFzQjtBQUNoRjtBQUNBO0FBQ0Esb0JBQW9CLDZDQUFnQjtBQUNwQztBQUNBO0FBQ0E7QUFLRTtBQUNGIiwic291cmNlcyI6WyIvVXNlcnMvaGFyc2hucGF0ZWwvRG9jdW1lbnRzL2dpdGh1Yi9Ob0NhcFJpZGUvY2xpZW50L25vZGVfbW9kdWxlcy8ucG5wbS9AcmFkaXgtdWkrcmVhY3QtZGlyZWN0aW9uQDEuMS4wX0B0eXBlcytyZWFjdEAxOS4wLjEwX3JlYWN0QDE5LjAuMC9ub2RlX21vZHVsZXMvQHJhZGl4LXVpL3JlYWN0LWRpcmVjdGlvbi9kaXN0L2luZGV4Lm1qcyJdLCJzb3VyY2VzQ29udGVudCI6WyIvLyBwYWNrYWdlcy9yZWFjdC9kaXJlY3Rpb24vc3JjL0RpcmVjdGlvbi50c3hcbmltcG9ydCAqIGFzIFJlYWN0IGZyb20gXCJyZWFjdFwiO1xuaW1wb3J0IHsganN4IH0gZnJvbSBcInJlYWN0L2pzeC1ydW50aW1lXCI7XG52YXIgRGlyZWN0aW9uQ29udGV4dCA9IFJlYWN0LmNyZWF0ZUNvbnRleHQodm9pZCAwKTtcbnZhciBEaXJlY3Rpb25Qcm92aWRlciA9IChwcm9wcykgPT4ge1xuICBjb25zdCB7IGRpciwgY2hpbGRyZW4gfSA9IHByb3BzO1xuICByZXR1cm4gLyogQF9fUFVSRV9fICovIGpzeChEaXJlY3Rpb25Db250ZXh0LlByb3ZpZGVyLCB7IHZhbHVlOiBkaXIsIGNoaWxkcmVuIH0pO1xufTtcbmZ1bmN0aW9uIHVzZURpcmVjdGlvbihsb2NhbERpcikge1xuICBjb25zdCBnbG9iYWxEaXIgPSBSZWFjdC51c2VDb250ZXh0KERpcmVjdGlvbkNvbnRleHQpO1xuICByZXR1cm4gbG9jYWxEaXIgfHwgZ2xvYmFsRGlyIHx8IFwibHRyXCI7XG59XG52YXIgUHJvdmlkZXIgPSBEaXJlY3Rpb25Qcm92aWRlcjtcbmV4cG9ydCB7XG4gIERpcmVjdGlvblByb3ZpZGVyLFxuICBQcm92aWRlcixcbiAgdXNlRGlyZWN0aW9uXG59O1xuLy8jIHNvdXJjZU1hcHBpbmdVUkw9aW5kZXgubWpzLm1hcFxuIl0sIm5hbWVzIjpbXSwiaWdub3JlTGlzdCI6WzBdLCJzb3VyY2VSb290IjoiIn0=\n//# sourceURL=webpack-internal:///(ssr)/./node_modules/.pnpm/@radix-ui+react-direction@1.1.0_@types+react@19.0.10_react@19.0.0/node_modules/@radix-ui/react-direction/dist/index.mjs\n");

/***/ })

};
;