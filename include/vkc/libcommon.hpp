#pragma once

#define VKC_NAMESPACE_NAME vkc
#define VKC_NAMESPACE_BEGIN namespace #VKC_NAMESPACE_NAME {
#define VKC_NAMESPACE_END };

#define VKC_API extern
/** to prevent multiple definitions error, default to static **/
#define VKC_INTRN static
