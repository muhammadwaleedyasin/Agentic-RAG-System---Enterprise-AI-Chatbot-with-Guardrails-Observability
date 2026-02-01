import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Routes that require authentication
const protectedRoutes = ['/chat', '/documents', '/dashboard', '/admin'];

// Routes that require admin role
const adminRoutes = ['/admin'];

// Routes that redirect to chat if authenticated
const authRoutes = ['/login'];

export function middleware(request: NextRequest) {
  const pathname = request.nextUrl.pathname;
  const token = request.cookies.get('auth_token')?.value;

  // Check if it's a protected route
  const isProtectedRoute = protectedRoutes.some(route => pathname.startsWith(route));
  const isAdminRoute = adminRoutes.some(route => pathname.startsWith(route));
  const isAuthRoute = authRoutes.some(route => pathname.startsWith(route));

  // If no token and trying to access protected route, redirect to login
  if (isProtectedRoute && !token) {
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('from', pathname);
    return NextResponse.redirect(loginUrl);
  }

  // If authenticated and trying to access auth routes, redirect to chat
  if (isAuthRoute && token) {
    return NextResponse.redirect(new URL('/chat', request.url));
  }

  // For admin routes, we do a basic check here
  // More sophisticated role checking happens client-side
  if (isAdminRoute && token) {
    try {
      // Basic JWT decode to check for admin role
      const payload = JSON.parse(atob(token.split('.')[1]));
      const hasAdminRole = payload.roles?.some((role: any) => 
        role.name === 'admin' || role.name === 'superuser'
      );
      
      if (!hasAdminRole) {
        return NextResponse.redirect(new URL('/dashboard', request.url));
      }
    } catch (error) {
      // If token is invalid, redirect to login
      return NextResponse.redirect(new URL('/login', request.url));
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public (public files)
     */
    '/((?!api|_next/static|_next/image|favicon.ico|public).*)',
  ],
};