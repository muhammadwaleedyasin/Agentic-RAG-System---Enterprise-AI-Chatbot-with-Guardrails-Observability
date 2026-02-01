'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useAuth } from '@/hooks/useAuth';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Loader2, AlertCircle } from 'lucide-react';

const loginSchema = z.object({
  username: z.string().min(1, 'Username is required'),
  password: z.string().min(1, 'Password is required'),
});

type LoginFormData = z.infer<typeof loginSchema>;

export default function LoginPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const { login } = useAuth();

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = async (data: LoginFormData) => {
    setIsLoading(true);
    setError(null);

    try {
      await login(data);
      router.push('/chat');
    } catch (err) {
      console.error('Login error:', err);
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-secondary-50 dark:bg-secondary-950 px-4">
      <Card className="w-full max-w-md p-6">
        <div className="text-center mb-6">
          <div className="w-12 h-12 bg-primary-600 rounded-lg flex items-center justify-center mx-auto mb-4">
            <span className="text-white font-bold text-lg">AR</span>
          </div>
          <h1 className="text-2xl font-bold text-secondary-900 dark:text-secondary-100">
            Welcome Back
          </h1>
          <p className="text-secondary-600 dark:text-secondary-400 mt-2">
            Sign in to your account to continue
          </p>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          {error && (
            <div
              className="bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800 rounded-md p-3 flex items-start gap-2"
              role="alert"
              aria-live="polite"
            >
              <AlertCircle className="h-4 w-4 text-error-600 dark:text-error-400 mt-0.5 flex-shrink-0" />
              <span className="text-sm text-error-800 dark:text-error-200">
                {error}
              </span>
            </div>
          )}

          <div className="space-y-2">
            <label
              htmlFor="username"
              className="block text-sm font-medium text-secondary-700 dark:text-secondary-300"
            >
              Username
            </label>
            <Input
              {...register('username')}
              id="username"
              type="text"
              autoComplete="username"
              disabled={isLoading}
              data-testid="login-username"
              aria-describedby={errors.username ? 'username-error' : undefined}
              aria-invalid={!!errors.username}
            />
            {errors.username && (
              <p
                id="username-error"
                className="text-sm text-error-600 dark:text-error-400"
                role="alert"
              >
                {errors.username.message}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <label
              htmlFor="password"
              className="block text-sm font-medium text-secondary-700 dark:text-secondary-300"
            >
              Password
            </label>
            <Input
              {...register('password')}
              id="password"
              type="password"
              autoComplete="current-password"
              disabled={isLoading}
              data-testid="login-password"
              aria-describedby={errors.password ? 'password-error' : undefined}
              aria-invalid={!!errors.password}
            />
            {errors.password && (
              <p
                id="password-error"
                className="text-sm text-error-600 dark:text-error-400"
                role="alert"
              >
                {errors.password.message}
              </p>
            )}
          </div>

          <Button
            type="submit"
            disabled={isLoading}
            className="w-full"
            data-testid="login-submit"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Signing in...
              </>
            ) : (
              'Sign In'
            )}
          </Button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-sm text-secondary-600 dark:text-secondary-400">
            Need help?{' '}
            <a
              href="#"
              className="text-primary-600 dark:text-primary-400 hover:underline focus:outline-none focus:underline"
            >
              Contact support
            </a>
          </p>
        </div>
      </Card>
    </div>
  );
}